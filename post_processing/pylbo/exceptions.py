class LegolasException(Exception):
    """
    Exception superclass to handle Legolas exceptions.

    Parameters
    ----------
    message : str
        The message to pass as error message.
    """

    def __init__(self, message=None):
        super().__init__(self, message)


class InvalidLegolasFile(LegolasException):
    """
    Handles trying to load invalid Legolas files.

    Parameters
    ----------
    file : str, ~os.PathLike
        The path to the file.
    """

    def __init__(self, file):
        self.file = file

    def __str__(self):
        return f"{self.file} is not a valid Legolas file"


class BackgroundNotPresent(LegolasException):
    """
    Handles trying to query for the background when this is not present in the datfile.

    Parameters
    ----------
    file : str, ~os.PathLike
        The path to the file.
    unable_to_do : str
        The thing that was unable to be done.
    """

    def __init__(self, file, unable_to_get=None):
        self.file = file
        self.unable_to_do = unable_to_get

    def __str__(self):
        msg = f"No background present in {self.file}"
        if self.unable_to_do is not None:
            msg = f"{msg}, unable to {self.unable_to_do}"
        return msg


class EigenfunctionsNotPresent(LegolasException):
    """
    Handles trying to query for eigenfunctions when these
    are not present in the datfile.

    Parameters
    ----------
    msg : str
        The error message to pass on.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"{self.msg}"


class MatricesNotPresent(LegolasException):
    """
    Handles trying to query for matrices when these
    are not present in the datfile.

    Parameters
    ----------
    file : str, ~os.PathLike
        The path to the file.
    """

    def __init__(self, file):
        self.file = file

    def __str__(self):
        return f"No matrices present in {self.file}"


class EigenvectorsNotPresent(LegolasException):
    """
    Handles trying to query for eigenvectors when
    these are not present in the datfile.

    Parameters
    ----------
    file : str, ~os.PathLike
        The path to the file.
    """

    def __init__(self, file):
        self.file = file

    def __str__(self):
        return f"No eigenvectors present in {self.file}"


class ResidualsNotPresent(LegolasException):
    """
    Handles trying to query for residuals when
    these are not present in the datfile.

    Parameters
    ----------
    file : str, ~os.PathLike
        The path to the file.
    """

    def __init__(self, file):
        self.file = file

    def __str__(self):
        return f"No residuals present in {self.file}"


class ParfileGenerationError(LegolasException):
    """
    Gets thrown when something went wrong during parfile generation.
    """

    def __init__(self, file, nb_runs=None, key=None):
        self.file = file
        self.nb_runs = nb_runs
        self.key = key

    def __str__(self):
        base_msg = "Inconsistency encountered during parfile generation! \n"
        if self.nb_runs is None:
            error_msg = (
                f"There are still some variables remaining in the supplied dictionary, "
                f"check keys/values. Remaining variables: \n"
                f"{self.file}"
            )
        else:
            error_msg = (
                f"Number of runs: {self.nb_runs} \n"
                f"Length of '{self.key}' key: {len(self.file.get(self.key))}"
            )
        return "".join([base_msg, error_msg])


class IVSnapshotsNotPresent(LegolasException):
    """
    Handles trying to query for initial value snapshots when
    these are not present in the datfile.
    """

    def __init__(self, file):
        self.file = file

    def __str__(self):
        return f"No initial value snapshots present in {self.file}"
