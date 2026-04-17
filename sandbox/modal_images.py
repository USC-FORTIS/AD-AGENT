from modal import Image

PYOD_IMAGE = Image.debian_slim(python_version="3.10").pip_install(
    "pyod", "numpy", "scikit-learn", "pandas", "scipy"
)

PYGOD_IMAGE = (
    Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy")
    .run_commands(
        "TORCH=$(python -c \"import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}.0')\") && "
        "pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster "
        "-f https://data.pyg.org/whl/torch-${TORCH}+cpu.html",
    )
    .pip_install("torch-geometric", "pygod")
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

TSB_AD_IMAGE = Image.debian_slim(python_version="3.10").pip_install(
    "TSB-AD", "torch", "numpy", "pandas", "scikit-learn", "scipy"
)

IMAGE_MAP = {
    "pyod": PYOD_IMAGE,
    "pygod": PYGOD_IMAGE,
    "darts": DARTS_IMAGE,
    "tslib": TSLIB_IMAGE,
    "tsb_ad": TSB_AD_IMAGE,
}
