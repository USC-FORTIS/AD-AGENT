import os

VALID_SANDBOX_MODES = ("modal", "docker")


def _load_sandbox_mode() -> str:
    """Resolve sandbox mode: env var > settings.yaml > default 'modal'."""
    # 1. Environment variable (highest priority)
    mode = os.environ.get("OPENAD_SANDBOX")
    if mode:
        if mode not in VALID_SANDBOX_MODES:
            raise ValueError(
                f"Invalid OPENAD_SANDBOX='{mode}'. Must be one of {VALID_SANDBOX_MODES}"
            )
        return mode

    # 2. settings.yaml fallback
    try:
        import yaml

        settings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "settings.yaml",
        )
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
        mode = settings.get("system", {}).get("sandbox_mode")
        if mode:
            if mode not in VALID_SANDBOX_MODES:
                raise ValueError(
                    f"Invalid sandbox_mode='{mode}' in settings.yaml. "
                    f"Must be one of {VALID_SANDBOX_MODES}"
                )
            return mode
    except (ImportError, FileNotFoundError):
        pass

    # 3. Default
    return "modal"


# "modal" uses Modal cloud sandboxes; "docker" uses local Docker containers
SANDBOX_MODE = _load_sandbox_mode()

MODAL_APP_NAME = "openad-sandbox"
MODAL_VOLUME_NAME = "openad-data"

# Default execution timeout in seconds
DEFAULT_TIMEOUT = 120
