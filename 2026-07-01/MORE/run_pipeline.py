import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
subprocess.check_call([sys.executable, str(root / "train.py")])
subprocess.check_call([sys.executable, str(root / "eval.py")])
