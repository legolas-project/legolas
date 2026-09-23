import warnings


def log_deprecation_warning(msg: str, since_version: str, stacklevel: int = 3) -> None:
    """
    Logs a deprecation warning.

    Parameters
    ----------
    msg : str
        The message to be displayed.
    since_version : str
        The version since which the deprecation warning is displayed.
    stacklevel : int, optional
        The stacklevel to be used for the warning, by default 3.
    """
    msg = "".join(
        [
            msg,
            f"\nDeprecated since version {since_version}. ",
            "May be removed in a future release.",
        ]
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
