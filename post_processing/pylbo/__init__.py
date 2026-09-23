import warnings

from pylbo._version import __version__
from pylbo.automation.api import generate_parfiles, run_legolas
from pylbo.utilities import logger
from pylbo.utilities.datfiles.file_loader import load, load_logfile, load_series
from pylbo.utilities.eq_balance import get_equilibrium_balance
from pylbo.utilities.logger import disable_logging, set_loglevel
from pylbo.visualisation.api import (
    plot_continua,
    plot_equilibrium,
    plot_equilibrium_balance,
    plot_matrices,
    plot_merged_spectrum,
    plot_spectrum,
    plot_spectrum_comparison,
    plot_spectrum_multi,
)
from pylbo.visualisation.modes.api import (
    plot_1d_temporal_evolution,
    plot_2d_slice,
    plot_3d_slice,
    prepare_vtk_export,
)
import pylbo.gimli

# allow deprecation warnings to be shown
warnings.filterwarnings("default", category=DeprecationWarning)

logger.init_logger()
