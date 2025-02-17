import time
import random
import logging
from collections import deque

class RateLimiter:
    """
    Professional rate limiter that supports:
      - Maximum requests per minute.
      - Maximum requests per hour.
      - Minimum time between requests.

    It also adds a random jitter factor to avoid exact synchronization in concurrent environments.
    """

    def __init__(
        self,
        max_requests_per_minute: int = None,
        max_requests_per_hour: int = None,
        min_time_between_requests: float = None,
        jitter_factor: float = 0.1
    ):
        """
        Initializes the rate limiter.

        Args:
            max_requests_per_minute (int): Maximum number of allowed requests per minute.
            max_requests_per_hour (int): Maximum number of allowed requests per hour.
            min_time_between_requests (float): Minimum time (in seconds) between requests.
            jitter_factor (float): Fraction of the waiting time to be used as randomness.
                                   For example, 0.1 adds +/-10% jitter.
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.min_time_between_requests = min_time_between_requests
        self.jitter_factor = jitter_factor

        self.minute_requests = deque()  # Timestamps for requests in the last 60 seconds.
        self.hour_requests = deque()    # Timestamps for requests in the last 3600 seconds.
        self.last_request_time = 0.0

        # Set up logging for debugging and information.
        self.logger = logging.getLogger(__name__)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        """

    def _cleanup(self, current_time: float):
        """
        Remove outdated request timestamps that fall outside the time windows.

        Args:
            current_time (float): The current time in seconds.
        """
        # Clean up timestamps older than 60 seconds.
        while self.minute_requests and current_time - self.minute_requests[0] > 60:
            removed = self.minute_requests.popleft()
            self.logger.debug(f"Removed minute timestamp: {removed}")

        # Clean up timestamps older than 3600 seconds (1 hour).
        while self.hour_requests and current_time - self.hour_requests[0] > 3600:
            removed = self.hour_requests.popleft()
            self.logger.debug(f"Removed hour timestamp: {removed}")

    def wait(self):
        """
        Waits until all rate-limiting conditions are satisfied,
        incorporating a random jitter to avoid exact synchronization.
        """
        while True:
            now = time.time()
            self._cleanup(now)

            wait_times = []

            # Constraint: Minimum time between requests.
            if self.min_time_between_requests is not None:
                elapsed = now - self.last_request_time
                if elapsed < self.min_time_between_requests:
                    wait_times.append(self.min_time_between_requests - elapsed)
                    self.logger.debug(f"Minimum time not met: need {self.min_time_between_requests - elapsed:.3f} more seconds.")

            # Constraint: Maximum requests per minute.
            if self.max_requests_per_minute is not None:
                if len(self.minute_requests) >= self.max_requests_per_minute:
                    remaining_minute = 60 - (now - self.minute_requests[0])
                    wait_times.append(remaining_minute)
                    self.logger.debug(f"Minute limit reached, wait {remaining_minute:.3f} seconds.")

            # Constraint: Maximum requests per hour.
            if self.max_requests_per_hour is not None:
                if len(self.hour_requests) >= self.max_requests_per_hour:
                    remaining_hour = 3600 - (now - self.hour_requests[0])
                    wait_times.append(remaining_hour)
                    self.logger.debug(f"Hour limit reached, wait {remaining_hour:.3f} seconds.")

            if wait_times:
                base_wait = max(wait_times)
                # Add random jitter: +/- up to jitter_factor * base_wait.
                jitter = random.uniform(-self.jitter_factor * base_wait, self.jitter_factor * base_wait)
                total_wait = base_wait + jitter
                total_wait = max(0, total_wait)  # Ensure non-negative waiting time.
                self.logger.info(f"Waiting {total_wait:.3f} seconds (base: {base_wait:.3f} s, jitter: {jitter:.3f} s).")
                time.sleep(total_wait)
            else:
                break

        # Record the current request.
        now = time.time()
        self.last_request_time = now
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.logger.debug("Request registered successfully.")
