import os

VALID_SANDBOX_MODES = ("modal", "docker")
SANDBOX_ENV_VAR = "ADAGENT_SANDBOX"
LEGACY_SANDBOX_ENV_VAR = "OPENAD_SANDBOX"
DEBUG_SANDBOX_ENV_VAR = "ADAGENT_SANDBOX_DEBUG"
LEGACY_DEBUG_SANDBOX_ENV_VAR = "OPENAD_SANDBOX_DEBUG"
MODAL_APP_ENV_VAR = "ADAGENT_MODAL_APP_NAME"
LEGACY_MODAL_APP_ENV_VAR = "OPENAD_MODAL_APP_NAME"
MODAL_VOLUME_ENV_VAR = "ADAGENT_MODAL_VOLUME_NAME"
LEGACY_MODAL_VOLUME_ENV_VAR = "OPENAD_MODAL_VOLUME_NAME"


def _load_sandbox_mode() -> str:
    """Resolve sandbox mode: env var > settings.yaml > default 'modal'."""
    # 1. Environment variable (highest priority)
    mode = os.environ.get(SANDBOX_ENV_VAR) or os.environ.get(LEGACY_SANDBOX_ENV_VAR)
    if mode:
        if mode not in VALID_SANDBOX_MODES:
            raise ValueError(
                f"Invalid {SANDBOX_ENV_VAR}='{mode}'. Must be one of {VALID_SANDBOX_MODES}"
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

MODAL_APP_NAME = (
    os.environ.get(MODAL_APP_ENV_VAR)
    or os.environ.get(LEGACY_MODAL_APP_ENV_VAR)
    or "adagent-sandbox"
)
MODAL_VOLUME_NAME = (
    os.environ.get(MODAL_VOLUME_ENV_VAR)
    or os.environ.get(LEGACY_MODAL_VOLUME_ENV_VAR)
    or "adagent-data"
)

# Default execution timeout in seconds
DEFAULT_TIMEOUT = 120
