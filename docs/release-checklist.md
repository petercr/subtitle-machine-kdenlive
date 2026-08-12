# v0.1.0 Release Checklist

`subtitle-machine-kdenlive` is a local desktop tool and MCP server. It has no
hosted service, database, or feature flags to deploy. A release is ready when
the locked Python package and the documented local-tool workflow are verified.

## Before tagging

- [ ] Confirm the intended release version in `pyproject.toml`.
- [ ] Confirm the release branch is clean and every intended PR is merged.
- [ ] Confirm the GitHub Actions matrix is green on Windows and Ubuntu.
- [ ] Run the locked test and compilation checks:

  ```powershell
  uv sync --locked --dev
  uv run --locked pytest
  uv run --locked python -m compileall -q src tests scripts
  ```

- [ ] Review `README.md`, `video-mcp.example.yaml`, and this checklist for
  paths, versions, and commands that changed during the release.

## Clean-machine validation

Perform this once on a Windows machine that does not reuse the development
virtual environment or existing project configuration.

1. Install Python 3.12 and `uv`, clone the tagged revision, then run
   `uv sync --locked --dev`.
2. Copy `video-mcp.example.yaml` to `video-mcp.yaml` and configure the local
   executable/model paths. Do not commit this machine-local file.
3. Check the environment:

   ```powershell
   uv run --locked video-mcp doctor
   uv run --locked python scripts/smoke_test.py --config video-mcp.yaml
   ```

4. Run one real video through the direct caption path. Verify that it produces
   a normalized transcript, SRT, ASS, preview, and editable Kdenlive project
   without modifying the source video:

   ```powershell
   uv run --locked video-mcp caption "C:\Videos\Test Video.mp4" --overwrite
   ```

5. If optional backends are enabled, verify each independently:

   - Whisper CUDA: `video-mcp doctor` reports both the executable and CUDA
     backend ready.
   - Parakeet: retain its experimental status and compare its output with
     Whisper on a representative clip.
   - Local LLM cleanup: start `llama-server` on `127.0.0.1`, configure
     `llm.server_url`, and confirm `video-mcp clean --json` reports
     `"used_llm": true`.

## Tag and publish

- [ ] Create an annotated `v0.1.0` tag from the verified `main` commit.
- [ ] Push the tag and create the GitHub release with the user-facing changes,
  setup requirements, and known optional-backend caveats.
- [ ] Attach no generated media, models, or machine-local configuration.

## Rollback and support

- If a packaged workflow fails, remove or mark the GitHub release as a
  pre-release and direct users to the previous verified tag.
- If a native backend fails, keep the deterministic path available and record
  the affected executable/model version in the issue or release notes.
- Do not change the default Whisper backend or make optional models required
  in a patch release without repeating the clean-machine validation.
