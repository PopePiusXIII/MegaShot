import os
import sys

# Ensure src is on sys.path for test imports.
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
