import os
import sys

# Ensure the repository root is first on sys.path so the workspace packages
# are loaded instead of any globally installed packages with the same name.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
