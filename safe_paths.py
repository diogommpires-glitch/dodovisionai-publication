"""Keep externally selected session files inside their owning directory."""

import re
from pathlib import Path


def safe_child(root: str | Path, name: str, suffix: str = "") -> Path:
    """Resolve one portable filename component, rejecting aliases and escapes."""
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10))
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[\w .-]+", name)
        or name in (".", "..")
        or name.endswith((" ", "."))
        or name.split(".", 1)[0].upper() in reserved
    ):
        raise ValueError("Session name must be one portable filename component")
    root = Path(root).resolve()
    path = (root / f"{name}{suffix}").resolve()
    if path.parent != root:
        raise ValueError("Session path must remain inside its storage directory")
    return path
