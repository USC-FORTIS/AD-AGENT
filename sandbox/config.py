import os

# "modal" uses Modal cloud sandboxes; "local" uses subprocess.run() (current behavior)
SANDBOX_MODE = os.environ.get("OPENAD_SANDBOX", "modal")

MODAL_APP_NAME = "openad-sandbox"

# Default execution timeout in seconds
DEFAULT_TIMEOUT = 120
