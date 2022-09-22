from pathlib import Path

def get_path_to(package) -> Path:
    """Get path to directory holding a package's __init__.py file"""
    return Path(package.__path__[0])