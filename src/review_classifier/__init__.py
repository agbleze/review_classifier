# read version from installed package
from importlib.metadata import version
import logging

package = __name__
__version__ = version(package)


logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)
logger.info(f"{package} version {__version__}")