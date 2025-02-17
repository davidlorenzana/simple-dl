import os
import io
import random
import logging
import hashlib
import zlib
import gzip
import json as json_
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Union, Tuple

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image
from urllib.parse import urlparse

from .rate_limiter import RateLimiter
from .utils import ensure_directory_exists

DEFAULT_USER_AGENT = "SimpleDownloader/1.0 (https://pypi.org/project/simple-dl/)"


class SimpleDownloader:
    """
    A simple downloader that supports caching (with SQLite), rate limiting,
    retries, and optional media (image) processing/compression.

    Args:
        db_filepath (str): Path to the SQLite database file for caching.
        max_age_days (Optional[int]): Cache expiration time in days. None means never expire.
        compress (bool): Whether to use ZLIB compression for cached content.
        compress_media (bool): Whether to apply lossless WebP compression to images.
        media_directory (str): Directory to save downloaded media files.
        max_requests_per_minute (Optional[int]): Maximum number of requests per minute.
        max_requests_per_hour (Optional[int]): Maximum number of requests per hour.
        min_time_between_requests (Optional[float]): Minimum time between requests in seconds.
        proxies (Optional[List[str]]): A list of proxy URLs.
        user_agents (Optional[List[str]]): A list of User-Agent strings.
        data_validator (Optional[Callable[[Any], bool]]): Callable to validate API responses.
        log_handler (Optional[Union[logging.Handler, str]]): Logging handler. If None, logging is suppressed;
            if "info", a StreamHandler at INFO level is used.
        generate_key (Optional[Callable[[str], str]]): Custom cache key generator.
    """

    def __init__(
        self,
        db_filepath: str = "scraped_data.db",
        max_age_days: Optional[int] = 7,
        compress: bool = False,
        compress_media: bool = False,
        media_directory: str = "downloads",
        max_requests_per_minute: Optional[int] = None,
        max_requests_per_hour: Optional[int] = None,
        min_time_between_requests: Optional[float] = None,
        proxies: Optional[List[str]] = None,
        user_agents: Optional[List[str]] = None,
        data_validator: Optional[Callable[[Any], bool]] = None,
        log_handler: Optional[Union[logging.Handler, str]] = None,
        generate_key: Optional[Callable[..., str]] = None,  # Custom key generator
    ):
        # Core configuration
        self.db_filepath = db_filepath
        self.max_age_days = max_age_days
        self.compress = compress
        self.compress_media = compress_media
        self.data_validator = data_validator
        self.generate_key = generate_key or self._default_generate_key

        # Media handling
        self.media_directory = media_directory

        # Request management
        self.proxies = proxies or []
        self.user_agents = user_agents or []
        self.session = self._create_session(self._get_random_user_agent())

        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=max_requests_per_minute,
            max_requests_per_hour=max_requests_per_hour,
            min_time_between_requests=min_time_between_requests,
        )

        # Logging setup
        self._logger = self._setup_logging(log_handler)

        # Database initialization (lazy)
        self._conn: Optional[sqlite3.Connection] = None
        self._tables_created: set[str] = set()
        ensure_directory_exists(db_filepath)

    def __enter__(self) -> "SimpleDownloader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close_connection()
        if exc_type is not None:
            self._logger.exception("An exception occurred within the context:")
            return False
        return True

    def close_connection(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # Public API Methods
    # -------------------------------------------------------------------------

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        force_refresh: bool = False,
        cache: bool = False,
        cache_table: str = "data",
        filepath: Optional[str] = None,
        parser: Optional[Union[str, Callable[[requests.Response], Any]]] = "text",
        serializer: Optional[Callable[[str], str]] = None,
        deserializer: Optional[Callable[[str], Any]] = None,
        compress: Optional[bool] = None,
        decompress: bool = False,
        **kwargs,
    ) -> Union[str, bytes, Any, bool, None]:
        """
        Retrieves web content from the specified URL with caching and parsing capabilities.

        If a file path is provided, the raw response is saved to disk.

        Returns:
            Parsed content, True if file saved, or None if an error occurred.
        """
        cache_key = self.generate_key(url, params=params, headers=headers) if cache else None

        if cache and not force_refresh:
            cached_content = self._get_content(cache_key, cache_table, deserializer, compress)
            if cached_content is not None and not self._should_refresh(cache_key, cache_table):
                self._logger.debug(f"Cache hit for {url}")
                return cached_content

        # Fetch fresh content
        response = self._fetch(
            "GET",
            url,
            cache_key=cache_key,
            cache_table=cache_table,
            params=params,
            headers=headers,
            cache=cache,
            **kwargs,
        )

        if filepath:
            try:
                self._save_content(filepath, response)
                self._logger.info(f"Content saved to {filepath}")
                return True
            except Exception as e:
                self._logger.error(f"Error saving to {filepath}: {e}")
                return None

        # Parse response content
        if parser is None:
            content = response.content
        elif parser == "text":
            content = response.text
        else:
            content = parser(response)

        if decompress and isinstance(content, bytes) and content.startswith(b"\x1f\x8b"):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
                    content = f.read()
                self._logger.debug(f"Content for {url} decompressed.")
            except gzip.BadGzipFile:
                self._logger.warning(f"Content for {url} appears gzipped but couldn't be decompressed.")

        if cache:
            self._store_content(cache_key, cache_table, content, serializer, compress)

        return content

    def query_api(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache: bool = False,
        cache_table: str = "data",
        **kwargs,
    ) -> Any:
        """
        Execute an API request with response caching.

        Returns:
            Parsed JSON (if valid) or raw text.
        """
        cache_key = self.generate_key(endpoint, method=method, params=params, json=json_data, headers=headers)

        if cache:
            cached = self._get_content(cache_key, cache_table)
            if cached is not None and not self._should_refresh(cache_key, cache_table):
                try:
                    return json_.loads(cached)
                except json_.JSONDecodeError:
                    return cached

        response = self._fetch(
            method,
            endpoint,
            cache_key=cache_key,
            cache_table=cache_table,
            params=params,
            json=json_data,
            headers=headers,
            cache=cache,
            **kwargs,
        )

        try:
            data = response.json()
            if self.data_validator:
                if self.data_validator(data):
                    if cache:
                        self._store_content(cache_key, cache_table, response.text)
                    return data
                else:
                    self._logger.warning(f"API response validation failed: {response.text[:200]}...")
                    return data
            else:
                if cache:
                    self._store_content(cache_key, cache_table, response.text)
                return data
        except json_.JSONDecodeError:
            self._logger.warning(f"Invalid JSON response: {response.text[:200]}...")
            return response.text

    def download_media(
        self,
        url: str,
        filename: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        chunk_size: int = 8192,
        store: bool = True,
    ) -> Optional[Union[str, io.BytesIO]]:
        """
        Downloads media from a URL. If the content is an image and lossless compression is enabled,
        the image is converted to WebP.

        Args:
            filename: Optional filename; if None, it is extracted from the URL.
            store: If True, the file is saved to disk; otherwise, a BytesIO object is returned.

        Returns:
            The file path (if stored) or a BytesIO object, or None if download fails.
        """
        if filename is None:
            filename = self._extract_filename_from_url(url) or "downloaded_file"
        filepath = os.path.join(self.media_directory, filename)

        if store and os.path.exists(filepath):
            return filepath

        blob, content_type = self._fetch_media(url, headers, chunk_size)
        if blob is None:
            return None

        if store:
            os.makedirs(self.media_directory, exist_ok=True)

        if content_type and content_type.startswith("image/"):
            return self._process_image(blob, filepath, store)
        else:
            if store:
                with open(filepath, "wb") as f:
                    f.write(blob)
                return filepath
            else:
                return io.BytesIO(blob)

    def iterate_table(
        self, table_name: str, decompress: Optional[bool] = None
    ) -> Any:
        """
        Iterate over cached table entries.

        Yields:
            Tuples of (key, content) from the specified table.
        """
        cursor = self.conn.execute(f"SELECT key, content FROM {table_name}")
        for key, content in cursor.fetchall():
            should_decompress = self.compress if decompress is None else decompress
            yield key, self._decompress_data(content) if should_decompress else content

    def cleanup(self, table_name: str = "data", max_days: int = 30) -> None:
        """
        Remove cached entries older than the specified number of days.
        """
        cutoff = datetime.now() - timedelta(days=max_days)
        with self.conn:
            self._check_table(table_name)
            self.conn.execute(
                f"DELETE FROM {table_name} WHERE created_at < ?",
                (cutoff.isoformat(),),
            )

    # Internal / Helper Methods
    # -------------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_filepath)
        return self._conn

    def _setup_logging(self, handler: Optional[Union[logging.Handler, str]]) -> logging.Logger:
        logger_ = logging.getLogger(__name__)
        logger_.setLevel(logging.DEBUG)
        logger_.handlers.clear()
        if handler is None:
            logger_.addHandler(logging.NullHandler())
        elif isinstance(handler, str) and handler == "info":
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            logger_.addHandler(ch)
        else:
            logger_.addHandler(handler)
        return logger_

    def _check_table(self, table_name: str) -> None:
        if table_name not in self._tables_created:
            self._create_table(table_name)

    def _create_table(self, table_name: str) -> None:
        with self.conn:
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    key TEXT PRIMARY KEY,
                    content BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    etag TEXT,
                    last_modified TEXT
                )
                """
            )
            self._tables_created.add(table_name)

    def _create_session(self, user_agent: str) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100, pool_block=True)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": user_agent})
        return session

    def _get_random_user_agent(self) -> str:
        return random.choice(self.user_agents) if self.user_agents else DEFAULT_USER_AGENT

    def _get_random_proxy(self) -> Optional[Dict[str, str]]:
        if self.proxies:
            proxy = random.choice(self.proxies)
            return {"http": proxy, "https": proxy}
        return None

    def _get_etag(self, key: str, table_name: str) -> Optional[str]:
        with self.conn:
            self._check_table(table_name)
            cursor = self.conn.execute(f"SELECT etag FROM {table_name} WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _store_etag(self, key: str, table_name: str, etag: Optional[str]) -> None:
        with self.conn:
            self._check_table(table_name)
            self.conn.execute(f"UPDATE {table_name} SET etag = ? WHERE key = ?", (etag, key))

    def _get_last_modified(self, key: str, table_name: str) -> Optional[str]:
        with self.conn:
            self._check_table(table_name)
            cursor = self.conn.execute(f"SELECT last_modified FROM {table_name} WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _store_last_modified(self, key: str, table_name: str, last_modified: Optional[str]) -> None:
        with self.conn:
            self._check_table(table_name)
            self.conn.execute(
                f"UPDATE {table_name} SET last_modified = ? WHERE key = ?",
                (last_modified, key),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _fetch(
        self,
        method: str,
        url: str,
        cache_key: Optional[str] = None,
        cache_table: Optional[str] = None,
        cache: bool = True,
        **kwargs,
    ) -> requests.Response:
        """
        Unified fetch method with rate limiting and retry logic.
        """
        self.rate_limiter.wait()

        headers: Dict[str, str] = kwargs.pop("headers", {}) or {}
        # Add caching headers if applicable
        if cache_key and cache_table:
            if etag := self._get_etag(cache_key, cache_table):
                headers["If-None-Match"] = etag
            elif last_modified := self._get_last_modified(cache_key, cache_table):
                headers["If-Modified-Since"] = last_modified

        # Rotate User-Agent and proxy
        headers["User-Agent"] = self._get_random_user_agent()
        if proxy := self._get_random_proxy():
            kwargs["proxies"] = proxy
        kwargs["headers"] = headers

        self._logger.debug(f"Fetching {method} {url}")
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()

        if response.status_code == 304 and cache_key is not None:
            self._logger.debug(f"Resource not modified (ETag/Last-Modified match): {url}")
            content = self._get_content(cache_key, cache_table)
            mock_response = requests.Response()
            mock_response.status_code = 304
            mock_response._content = (
                content.encode("utf-8") if isinstance(content, str) else content
            )
            return mock_response

        if response.status_code == 200 and cache and cache_key and cache_table:
            self._store_etag(cache_key, cache_table, response.headers.get("ETag"))
            self._store_last_modified(cache_key, cache_table, response.headers.get("Last-Modified"))
        return response

    def _default_generate_key(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> str:
        """
        Generate a SHA256 hash based on request parameters.
        """
        headers_to_cache = dict(headers) if headers else {}
        for volatile in ("User-Agent", "Proxy", "Cookie"):
            headers_to_cache.pop(volatile, None)

        request_metadata = {
            "url": url,
            "method": method,
            "params": params or {},
            "json": json or {},
            "headers": headers_to_cache,
        }
        input_data = json_.dumps(request_metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(input_data.encode()).hexdigest()

    def _compress_data(self, data: str) -> bytes:
        return zlib.compress(data.encode("utf-8"), level=3)

    def _decompress_data(self, data: bytes) -> str:
        return zlib.decompress(data).decode("utf-8")

    def _store_content(
        self,
        key: str,
        table_name: str,
        content: Union[str, bytes],
        serializer: Optional[Callable[[str], str]] = None,
        compress: Optional[bool] = None,
    ) -> None:
        if serializer:
            content = serializer(content)  # type: ignore

        use_compression = self.compress if compress is None else compress
        processed = self._compress_data(content) if use_compression and isinstance(content, str) \
            else content if isinstance(content, bytes) else content.encode("utf-8")

        with self.conn:
            self._check_table(table_name)
            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO {table_name}
                (key, content, created_at)
                VALUES (?, ?, ?)
                """,
                (key, processed, datetime.now().isoformat()),
            )

    def _get_content(
        self,
        key: str,
        table_name: str,
        deserializer: Optional[Callable[[str], Any]] = None,
        compress: Optional[bool] = None,
    ) -> Optional[str]:
        with self.conn:
            self._check_table(table_name)
            cursor = self.conn.execute(
                f"SELECT content FROM {table_name} WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        try:
            use_compression = self.compress if compress is None else compress
            data = (
                self._decompress_data(row[0])
                if use_compression else row[0].decode("utf-8")
            )
            return deserializer(data) if deserializer else data
        except (zlib.error, UnicodeDecodeError) as e:
            self._logger.error(f"Content processing failed: {e}")
            return None

    def _should_refresh(self, key: str, table_name: str) -> bool:
        if self.max_age_days is None:
            return False

        with self.conn:
            self._check_table(table_name)
            cursor = self.conn.execute(
                f"SELECT created_at FROM {table_name} WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

        if not row:
            return True

        expiration = datetime.now() - timedelta(days=self.max_age_days)
        return datetime.fromisoformat(row[0]) < expiration

    def _save_content(self, filepath: str, response: requests.Response) -> None:
        self._logger.info(f"Saving response content to {filepath}.")
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def _fetch_media(self, url: str, headers: Optional[Dict[str, str]] = None, chunk_size: int = 8192) -> Tuple[Optional[bytes], Optional[str]]:
        try:
            response = self._fetch(
                "GET",
                url,
                cache_key=self.generate_key(f"WEB::{url}"),
                headers=headers,
                stream=True,
                timeout=15,
            )
            if response.status_code != 200:
                self._logger.error(f"Download failed with status code: {response.status_code}")
                return None, None

            data = bytearray()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    data.extend(chunk)
            content_type = response.headers.get("Content-Type")
            return bytes(data), content_type
        except requests.exceptions.RequestException as e:
            self._logger.exception(f"Network error during download: {e}")
            return None, None

    def _process_image(
        self, blob: bytes, filepath: str, store: bool
    ) -> Optional[Union[str, io.BytesIO]]:
        try:
            with Image.open(io.BytesIO(blob)) as img:
                img_format = img.format.lower()
                if img_format in ("png", "tiff", "webp") and self.compress_media:
                    compressed_blob = io.BytesIO()
                    img.save(compressed_blob, "webp", lossless=True)
                    if store:
                        out_path = f"{filepath}.webp"
                        with open(out_path, "wb") as f:
                            f.write(compressed_blob.getvalue())
                        return out_path
                    else:
                        return compressed_blob
                else:
                    self._logger.info(
                        "JPEG image: Keeping original file." if img_format == "jpeg"
                        else f"No lossless compression for format: {img_format}"
                    )
                    if store:
                        with open(filepath, "wb") as f:
                            f.write(blob)
                        return filepath
                    else:
                        return io.BytesIO(blob)
        except Exception as e:
            self._logger.error(f"Error during image processing: {e}")
            if store:
                with open(filepath, "wb") as f:
                    f.write(blob)
                return filepath
            else:
                return io.BytesIO(blob)

    def _extract_filename_from_url(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            return os.path.basename(parsed.path)
        except Exception as e:
            self._logger.error(f"Error extracting filename from URL: {e}")
            return None
