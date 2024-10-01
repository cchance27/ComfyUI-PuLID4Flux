import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.realpath(__file__)))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]