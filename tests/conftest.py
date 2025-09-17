# This file places the "code" directory on sys.path.

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
