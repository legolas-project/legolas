import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# Search for Pylbo in $LEGOLASDIR/post_processing.
if "LEGOLASDIR" in os.environ.keys():
    _pylbo_path = Path(os.environ["LEGOLASDIR"]).joinpath("post_processing")
    if _pylbo_path.is_dir():
        sys.path.append(str(_pylbo_path.resolve()))

# Else search for Pylbo in ./post_processing, if possible.
elif "__file__" in globals():
    _pylbo_path = Path(__file__).parent.joinpath("post_processing")
    if _pylbo_path.is_dir():
        sys.path.append(str(_pylbo_path.resolve()))

try:
    import pylbo
except ModuleNotFoundError as e:
    print("ERROR: Failed to load Pylbo")
    print("      ", e.args[0])
    print()
    print(
        "Check if $LEGOLASDIR is configured correctly and all of Pylbo's",
        "dependencies are available, or install Pylbo as a package.",
    )
    exit(1)


def _main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--datfile", dest="datfile")
    args = parser.parse_args()
    datfile = args.datfile

    if datfile is None:
        datfile = Path("output/datfile.dat").resolve()
        if not datfile.is_file():
            raise FileNotFoundError(datfile)

    ds = pylbo.load(datfile)
    pylbo.plot_equilibrium(ds)
    p = pylbo.plot_spectrum(ds, use_residuals=ds.has_residuals)
    p.add_continua()
    if ds.has_efs:
        p.add_eigenfunctions()
    p.show()


if __name__ == "__main__":
    _main()
