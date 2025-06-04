import os
import shutil
from pathlib import Path


def mkdir(path: Path, exist_ok=True, delete_existing=False, mk_parents=True):
    """
    Creates the specified directory.

    Args:
        path (pathlib.Path): Path to be created
        exist_ok (bool: default True): If false, raises an error if the directory being created already exists.
        delete_existing (bool: default False): If true, deletes the previously existing directory before creating the new one.
        mk_parents (bool: default True): If true, recursively creates all parent directories required by path.

    Returns:
        None
    """
    if path.exists() and delete_existing:
        shutil.rmtree(path)

    path.mkdir(exist_ok=exist_ok, parents=mk_parents)