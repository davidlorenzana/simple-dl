import os
import pathlib

def ensure_directory_exists(filepath):
    """Ensures that the directory portion of a filepath exists.

    Args:
        filepath: The full filepath (including filename).
    """
    directory = os.path.dirname(filepath)  # Get the directory part of the path
    if not directory: #Handle the case where filepath is just a filename
        return
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
