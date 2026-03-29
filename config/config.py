import os

class Config:
    OPENAI_API_KEY = 'sk-REDACTED'

    # 'sk-REDACTED'

    # Sandbox configuration
    SANDBOX_MODE = os.environ.get("OPENAD_SANDBOX", "modal")  # "modal" | "docker"
    MODAL_APP_NAME = "openad-sandbox"
    SANDBOX_DEFAULT_TIMEOUT = 120
