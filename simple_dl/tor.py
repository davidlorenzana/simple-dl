import requests
from stem import Signal
from stem.control import Controller
import random
from typing import List, Optional, Dict

from .simple_dl import SimpleDownloader

class TorDownloader(SimpleDownloader):
    def __init__(
        self,
        tor_control_ports: Optional[List[int]] = None,
        tor_proxy: str = "socks5h://127.0.0.1:9050",
        *args,
        **kwargs
    ):
        """
        Initializes the TorDownloader subclass.

        Args:
            tor_control_ports: A list of Tor control ports for different Tor instances.
            tor_proxy: The SOCKS5 proxy address for Tor. Defaults to "socks5h://127.0.0.1:9050".
            *args: Arguments passed to the parent class (SimpleDownloader).
            **kwargs: Keyword arguments passed to the parent class (SimpleDownloader).
        """

        self.tor_control_ports = tor_control_ports or [9050]  # Default to port 9051
        self.tor_proxy = tor_proxy

        super().__init__(*args, proxies=[tor_proxy], **kwargs)

    def _get_random_proxy(self) -> Optional[Dict[str, str]]:
        """Selects a random proxy from the list."""
        port = random.choice(self.tor_control_ports)
        proxy = self.tor_proxy.replace("9050", str(port))
        return {"http": proxy, "https": proxy}  # Use the same proxy for both http and https

    def renew_tor_ip(self, control_port: Optional[int] = None, hashed_password: Optional[str] = None):
        """
        Renews the Tor IP address by requesting a new circuit for a specific control port.

        Args:
            control_port: The Tor control port to use. If None, a random port from the list is selected.
            hashed_password: The hashed control password for authentication if needed.
        """
        if control_port is None:
            control_port = random.choice(self.tor_control_ports)

        try:
            with Controller.from_port(port=control_port) as controller:
                if hashed_password:
                    controller.authenticate(password=hashed_password)
                else:
                    controller.authenticate()
                controller.signal(Signal.NEWNYM)
                self._logger.info(f"Tor IP address renewed using control port {control_port}.")
        except Exception as e:
            self._logger.error(f"Failed to renew Tor IP address using control port {control_port}: {e}")
