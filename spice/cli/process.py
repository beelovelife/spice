"""Process title support."""


def set_process_title() -> None:
    """Set the process title to 'spice' when the platform supports it."""
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle("spice")
        return
    except ImportError:
        pass

    import ctypes
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"spice", 0, 0, 0)
        elif system == "Darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            libc.pthread_setname_np(b"spice")
    except Exception:
        pass
