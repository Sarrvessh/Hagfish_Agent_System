# Utility helpers for the adaptive_trainer package


def as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default
