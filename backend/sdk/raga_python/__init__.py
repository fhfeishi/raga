

from beartype.claw import beartype_this_package
beartype_this_package()

import importlib.metadata
__version__ = importlib.metadata.version("raga_python")







