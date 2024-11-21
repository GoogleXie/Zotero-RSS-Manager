import sys
import pathlib
import types
# Ensure pathlib is available
if "pathlib" not in sys.modules:
    sys.modules["pathlib"] = pathlib

# Handle other imports gracefully
try:
    import importlib_resources.trees
except ImportError:
    sys.modules["importlib_resources.trees"] = types.ModuleType("importlib_resources.trees")  # Mock if not found

try:
    from notebook.services import shutdown
except ImportError:
    sys.modules["notebook.services.shutdown"] = types.ModuleType("notebook.services.shutdown")  # Mock if not found

# Mock egenix-mx-base if required
if "egenix-mx-base" not in sys.modules:
    sys.modules["mx.DateTime"] = type("Mock", (), {"DateTime": None})
