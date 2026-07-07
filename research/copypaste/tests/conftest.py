"""Make the `neardup` package importable when running pytest from anywhere."""
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
