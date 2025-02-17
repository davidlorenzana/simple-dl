# simple_dl

**`simple_dl`** is a Python module for fetching and caching web content. It supports rate limiting, retries, response caching via SQLite, and even optional lossless compression for image media. This makes it a great tool for web scraping, API querying, or downloading media files reliably.

## Features

- **Caching:** Cache responses in a SQLite database with configurable expiration and optional ZLIB compression.
- **Rate Limiting:** Prevent overloading servers by configuring maximum requests per minute/hour.
- **Retries:** Built-in retry mechanism with exponential backoff using [tenacity](https://github.com/jd/tenacity).
- **Media Processing:** Download and, optionally, compress image files using lossless WebP conversion.
- **Flexible API:** Easily configure user agents, proxies, and custom cache key generators.
- **Logging:** Built-in logging support with customizable logging handlers.
- **Tor Support:** The `TorDownloader` class allows you to use the Tor network for requests, with the ability to renew your Tor IP address.

## Installation

To install `simple_dl` from source, clone the repository and run:

```bash
git clone https://github.com/davidlorenzana/simple-dl.git
cd simple-dl
pip install .
```

If you want to use the Tor functionality, install the package with the tor extra:

```bash
pip install .[tor]
```

## Usage

Below are a few examples to help you get started.

### Basic Content Download

```python
from simple_dl import SimpleDownloader

# Create a downloader instance with info logging
downloader = SimpleDownloader(log_handler="info")

url = "http://example.com"
content = downloader.get(url)

print(content)
```

### Using Caching

```python
from simple_dl import SimpleDownloader

# Enable caching (responses are stored in a SQLite database)
downloader = SimpleDownloader(log_handler="info")

url = "http://example.com"
# Set cache=True to cache the response for subsequent requests.
content = downloader.get(url, cache=True)

print(content)
```

### Downloading Media (Images)

```python
from simple_dl import SimpleDownloader

downloader = SimpleDownloader(log_handler="info")

# Download an image from the provided URL. If the image is in a lossless format,
# and if compress_media is enabled, it will be converted to WebP.
media_filepath = downloader.download_media("http://example.com/image.png")

print(f"Media saved to: {media_filepath}")
```

### Querying APIs with Caching and Validation

```python
from simple_dl import SimpleDownloader

def validate_response(data):
    # Implement your validation logic here (e.g., check if a key exists)
    return "expected_key" in data

downloader = SimpleDownloader(log_handler="info", data_validator=validate_response)

api_endpoint = "http://api.example.com/data"
response_data = downloader.query_api(api_endpoint, cache=True)

print(response_data)
```

### Using Tor with TorDownloader

The `TorDownloader` class extends `SimpleDownloader` to enable routing requests through the Tor network. This is especially useful if you need to hide your IP address or access content via Tor nodes.

```python
from simple_dl import TorDownloader

# Initialize the Tor downloader.
# You can specify a list of Tor control ports if you are running multiple Tor instances.
tor_downloader = TorDownloader(
    tor_control_ports=[9050, 9051],
    tor_proxy="socks5h://127.0.0.1:9050",
    log_handler="info"
)

# Make a request through Tor.
url = "http://example.com"
content = tor_downloader.get(url)
print(content)

# To renew the Tor IP address (i.e., request a new circuit), call:
tor_downloader.renew_tor_ip()
```

## Configuration Options

When creating a `SimpleDownloader` instance, you can configure:

- **Database & Caching:**
  - `db_filepath`: SQLite file path for caching.
  - `max_age_days`: Expiration (in days) for cached responses (set to `None` to never expire).
  - `compress`: Enable ZLIB compression for cached text data.

- **Media Handling:**
  - `compress_media`: If enabled, lossless image formats (e.g., PNG, TIFF) are converted to WebP.
  - `media_directory`: Directory to store downloaded media files.

- **Request Management:**
  - `max_requests_per_minute` / `max_requests_per_hour` / `min_time_between_requests`: For rate limiting.
  - `proxies`: List of proxy URLs.
  - `user_agents`: List of user agent strings for rotation.

- **Logging:**
  - `log_handler`: Provide `"info"` for a default StreamHandler, `None` for no logging, or a custom `logging.Handler` instance.

- **Custom Cache Key:**
  - `generate_key`: Provide your own function to generate a cache key from request parameters.

- **Tor Support (Only in TorDownloader):**

  - `tor_control_ports`: List of Tor control ports (default is [9050] or [9050, 9051] if using multiple instances).
  - `tor_proxy`: SOCKS5 proxy address for Tor (default is "socks5h://127.0.0.1:9050").

## Contributing

Contributions are welcome! Please open issues and pull requests on the [GitHub repository](https://github.com/yourusername/simple-downloader).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
