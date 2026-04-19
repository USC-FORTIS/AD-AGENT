import os


class Config:
    OPENAI_API_KEY = ''

    SANDBOX_MODE = os.environ.get("ADAGENT_SANDBOX") or os.environ.get("OPENAD_SANDBOX", "modal")
    MODAL_APP_NAME = os.environ.get("ADAGENT_MODAL_APP_NAME") or os.environ.get("OPENAD_MODAL_APP_NAME") or "adagent-sandbox"
    SANDBOX_DEFAULT_TIMEOUT = 120
