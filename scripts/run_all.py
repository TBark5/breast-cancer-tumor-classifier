"""Run pipeline, documentation, and tests; propagate every failure."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "src.workflow"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/build_readme.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
