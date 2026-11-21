"""
DODO VISION AI — Utilities
"""
import numpy as np

def sanitize(o):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    elif isinstance(o, (np.integer,)):
        return int(o)
    elif isinstance(o, (np.floating,)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    return o
