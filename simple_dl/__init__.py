from .simple_dl import SimpleDownloader

try:
    from .tor import TorDownloader
    tor_support = True
except ImportError as e:
    tor_support = False

