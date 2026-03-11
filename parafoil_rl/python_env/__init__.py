# Makes python_env/ a package and ensures parafoil_cpp.pyd is on sys.path
import sys
from pathlib import Path

_here = Path(__file__).parent.resolve()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
