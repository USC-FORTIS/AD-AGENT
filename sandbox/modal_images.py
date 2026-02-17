from modal import Image

PYOD_IMAGE = Image.debian_slim(python_version="3.10").pip_install(
    "pyod", "numpy", "scikit-learn", "pandas"
)

PYGOD_IMAGE = Image.debian_slim(python_version="3.10").pip_install(
    "pygod", "torch", "torch-geometric", "numpy"
)

DARTS_IMAGE = Image.debian_slim(python_version="3.10").pip_install(
    "darts", "torch", "numpy", "pandas"
)

TSLIB_IMAGE = (
    Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "pandas", "scikit-learn")
    .run_commands(
        "apt-get update && apt-get install -y git",
        "git clone https://github.com/thuml/Time-Series-Library.git /opt/Time-Series-Library",
    )
)

IMAGE_MAP = {
    "pyod": PYOD_IMAGE,
    "pygod": PYGOD_IMAGE,
    "darts": DARTS_IMAGE,
    "tslib": TSLIB_IMAGE,
}
