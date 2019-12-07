import imageio
import warnings

warnings.simplefilter("ignore", UserWarning)
imageio.plugins.freeimage.download()

from .core import *
from .data import *
