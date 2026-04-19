# Sandbox Notes

The canonical sandbox configuration note lives at:

- `docs/notes/docker-sandbox-summary.md`

That document covers:

- backend selection (`modal` vs `docker`)
- `ADAGENT_SANDBOX` and `ADAGENT_SANDBOX_DEBUG`
- Modal app and volume behavior
- Docker image build and mount behavior
- current logging and runtime caveats

Implementation entry points in this package:

- `config.py`
- `executor.py`
- `docker_images.py`
- `modal_images.py`
