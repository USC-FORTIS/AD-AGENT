import os


class Config:
    OPENAI_API_KEY = ''

    SANDBOX_MODE = os.environ.get("OPENAD_SANDBOX", "modal")
    MODAL_APP_NAME = "openad-sandbox"
    SANDBOX_DEFAULT_TIMEOUT = 120
