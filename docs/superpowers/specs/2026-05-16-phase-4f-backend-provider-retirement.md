# Phase 4f — Retire `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` dials

**Status:** In progress.
**Tracks:** epic #337.
**Predecessor:** Phase 4e (PR #379, merged) — legacy concrete handlers deleted; `FACTORY_PATH` retired. Phase 4d (PR #378) flipped the composable factory to default. After 4e, both legacy dials (`BACKEND_PROVIDER` + `LOCAL_STT_RESPONSE_BACKEND`) are still referenced across 5 surfaces but no longer drive the runtime path — they are pure cruft.

## Why this is the final 4f sub-phase

Per PIPELINE_REFACTOR.md §"Sub-phase 4f scope discovery (2026-05-16)", the
original 4f brief (`config.py` + `main.py` + `.env.example` + `profiles/` +
`deploy/`) was too narrow: post-4e there are **5 distinct surfaces** still
reading `config.BACKEND_PROVIDER` / `config.LOCAL_STT_RESPONSE_BACKEND`.
The operator authorised **Option B: ship as one PR** (~600-1000 LOC, 5-6
commits) with explicit approvals for:

- Admin UI JSON contract change (status payload + `/backend_config` POST shape).
- Renaming the Python ClassVar (`BACKEND_PROVIDER` → `PROVIDER_ID`) without
  touching the *emitted* OTel attribute *value* strings (`"huggingface"`,
  `"openai"`, `"gemini"`, `"llama_cpp"`).
- No migration logic for operator `.env` files — `BACKEND_PROVIDER=...` rows
  become ignored after deploy; operators must re-save from the admin UI.

## The 5 surfaces and the retirement fix per surface

### Surface 1 — `src/robot_comic/prompts.py` (`_filter_delivery_tags`)

Reads `config.BACKEND_PROVIDER` + `config.LOCAL_STT_RESPONSE_BACKEND` to
decide whether to keep or strip the `## GEMINI TTS DELIVERY TAGS` section
in the system prompt.

**Fix:** switch the gate to `config.AUDIO_OUTPUT_BACKEND == AUDIO_OUTPUT_GEMINI_TTS`.
The two existing legacy values `GEMINI_TTS_OUTPUT` and `LLAMA_GEMINI_TTS_OUTPUT`
both map to the same canonical `AUDIO_OUTPUT_GEMINI_TTS` (see
`_LOCAL_STT_RESPONSE_TO_AUDIO_OUTPUT` in config.py), so the new gate
preserves the semantic of "the active TTS renders Gemini delivery tags."

`_uses_gemini_tts` is simplified to take a single argument
`output_backend: str` and return `output_backend == AUDIO_OUTPUT_GEMINI_TTS`.

### Surface 2 — `src/robot_comic/base_realtime.py` + handler subclasses

`BACKEND_PROVIDER: ClassVar[str]` is declared on `BaseRealtimeHandler` and
set on each concrete realtime subclass. It serves two purposes:

1. **Voice catalog lookup** — passed to `get_available_voices_for_backend()` /
   `get_default_voice_for_backend()`.
2. **OTel attributes** — `gen_ai.system` (5 sites) and `robot.mode` (2 sites)
   carry the ClassVar value into telemetry.

**Fix:**

- Rename `BACKEND_PROVIDER` → `PROVIDER_ID` on `BaseRealtimeHandler` and on
  every concrete subclass (`HuggingFaceRealtimeHandler`,
  `OpenaiRealtimeHandler`, `LocalSTTOpenAIRealtimeHandler`,
  `LocalSTTHuggingFaceRealtimeHandler`).
- The hybrid `LocalSTT*RealtimeHandler` classes already override
  `BACKEND_PROVIDER` to the *response* backend (OPENAI_BACKEND / HF_BACKEND)
  rather than `LOCAL_STT_BACKEND`. That semantic is preserved — they now
  override `PROVIDER_ID` instead.
- Voice-catalog helper renames:
  - `get_available_voices_for_backend(backend_provider)` →
    `get_available_voices_for_provider(provider_id)` (semantic unchanged;
    `provider_id` ∈ {`openai`, `gemini`, `huggingface`, `local_stt`}).
  - `get_default_voice_for_backend(backend_provider)` →
    `get_default_voice_for_provider(provider_id)`.
  - The OTel emission *values* (`gen_ai.system="huggingface"`, etc.) stay
    identical — only the Python field name changes.

### Surface 3 — `src/robot_comic/console.py` admin UI

`_persist_backend_choice(backend)` writes `BACKEND_PROVIDER=<value>` into the
instance `.env`. `/backend_config` POST accepts a `backend` field plus
`local_stt_response_backend`. `_status_payload` emits `backend_provider` and
`local_stt_response_backend`.

**Fix:** reformulate around the new dimensions.

**New `/backend_config` POST payload (validated by Pydantic):**

| Field                  | Type             | Required | Notes                                          |
|------------------------|------------------|----------|------------------------------------------------|
| `pipeline_mode`        | str              | yes      | One of `PIPELINE_MODE_CHOICES`                 |
| `audio_input_backend`  | str \| None      | when composable | One of `AUDIO_INPUT_CHOICES`           |
| `audio_output_backend` | str \| None      | when composable | One of `AUDIO_OUTPUT_CHOICES`          |
| `llm_backend`          | str \| None      | optional | `llama` (default) or `gemini`                  |
| `api_key`              | str \| None      | optional | OpenAI / Gemini key — only when relevant       |
| `hf_mode` / `hf_host` / `hf_port` | …      | optional | HF connection details                          |
| `elevenlabs_api_key`, `elevenlabs_voice` | … | optional | ElevenLabs creds                              |
| `local_stt_*`          | …                | optional | Moonshine settings, unchanged                  |

The legacy fields `backend` and `local_stt_response_backend` are removed.

**New `/status` JSON shape (additive + removals):**

- **Added:** `pipeline_mode`, `pipeline_mode_choices`,
  `audio_input_backend`, `audio_input_backend_choices`,
  `audio_output_backend`, `audio_output_backend_choices`,
  `active_pipeline_mode`.
- **Removed:** `backend_provider`, `local_stt_response_backend`,
  `local_stt_response_backend_choices`.
- Existing fields kept: `active_backend`, `has_key`, `has_*_key`,
  `can_proceed*`, `hf_*`, `local_stt_provider`, `llm_backend`,
  `crowd_history_*`, `requires_restart`, etc.

The `active_backend` field is preserved because it's a useful coarse-grained
status indicator ("is the *running* handler still the right kind?"). It is
now derived from the *active pipeline mode + output backend* via a new
helper `provider_id_from_pipeline(pipeline_mode, audio_output_backend)`.

**Persisted-`.env` writes by `_persist_pipeline_choice(...)`:**

- `REACHY_MINI_PIPELINE_MODE=<mode>`
- For composable mode only: `REACHY_MINI_AUDIO_INPUT_BACKEND=<...>` +
  `REACHY_MINI_AUDIO_OUTPUT_BACKEND=<...>` + `REACHY_MINI_LLM_BACKEND=<...>`
- Old keys (`BACKEND_PROVIDER=...`, `LOCAL_STT_RESPONSE_BACKEND=...`) are
  no longer written. Existing rows in the operator's `.env` from a previous
  install simply become ignored on next reload.

### Surface 4 — `src/robot_comic/static/main.js` + `index.html`

The static admin UI's radio buttons are conceptually grouped by 4 "pipeline
families": Hugging Face, OpenAI Realtime, Gemini Live, Local STT. The
underlying 3-column STT/LLM/TTS picker for the Local STT family is already
wired (#245 / `test_admin_pipeline_3column.py`). The HTML structure does
not change — only the JavaScript serialization to/from the server.

**JS changes:**

- Replace reads of `status.backend_provider` with
  `pipelineModeToFamily(status.pipeline_mode, status.audio_output_backend)`
  (a new tiny helper).
- Replace writes of `body.backend` / `body.local_stt_response_backend` with
  `body.pipeline_mode` / `body.audio_input_backend` /
  `body.audio_output_backend` (computed from the family + 3-column
  selections).
- The radio button `value` attribute on each family card stays unchanged
  (`huggingface`, `openai`, `gemini`, `local_stt`) — it's purely
  presentational client-side state now.

### Surface 5 — originally-anticipated cleanup

- `src/robot_comic/config.py`:
  - Delete `BACKEND_PROVIDER` and `LOCAL_STT_RESPONSE_BACKEND` from `Config`
    and `refresh_runtime_config_from_env`.
  - Delete `DEFAULT_BACKEND_PROVIDER`, `_normalize_backend_provider`,
    `_normalize_local_stt_response_backend`, `_resolve_model_name`.
  - Replace `derive_audio_backends` callers: the function is still useful as
    a derivation primitive, but takes a `provider_id` string instead of a
    "backend provider" config value. Or fold it entirely if it becomes
    unused after surface 3.
  - Delete `_LOCAL_STT_RESPONSE_TO_AUDIO_OUTPUT` if orphaned.
  - Delete `LOCAL_STT_RESPONSE_BACKEND_ENV`, `LOCAL_STT_RESPONSE_BACKEND_CHOICES`,
    `LOCAL_STT_DEFAULT_RESPONSE_BACKEND`.
  - Delete `resolve_audio_backends` if orphaned (the env-var override path
    is now the *only* path).
  - Rewrite `get_backend_choice` → `get_provider_id` (active provider for
    catalog lookups) and `get_backend_label` to compute from
    `PIPELINE_MODE` + `AUDIO_OUTPUT_BACKEND` instead of the dial.
- `src/robot_comic/main.py:240-251` — remove the HF-specific logging fork;
  emit a single `logger.info("Configured pipeline mode: %s, ...", ...)` line.
- `.env.example` — remove the `BACKEND_PROVIDER=…` row and the
  `LOCAL_STT_RESPONSE_BACKEND=…` row; add a brief pointer to
  `REACHY_MINI_PIPELINE_MODE` for operators reading the file.
- `deploy/`, `profiles/`, `pyproject.toml` — verified clean by the prior
  agent (grep returns zero matches); re-verify before push.
- Retirement-explainer comment in `config.py` (same pattern Phase 4e used
  for `FACTORY_PATH`): a one-line stub noting the dials were retired in 4f
  and pointing at this spec.

## Acceptance criteria

1. `git grep "BACKEND_PROVIDER\|LOCAL_STT_RESPONSE_BACKEND" -- src/ profiles/ deploy/ .env.example pyproject.toml tests/`
   returns NOTHING. (One retirement-explainer comment in `config.py` is
   permitted — same pattern Phase 4e used.)
2. `uvx ruff@0.12.0 check` and `uvx ruff@0.12.0 format --check` both green
   from the repo root.
3. `.venv/bin/mypy --pretty <changed files>` green.
4. `pytest tests/ -q --ignore=tests/vision/test_local_vision.py` green.
5. The app boot path on the robot still works (no manual hardware test —
   reasoned through `main.py` post-cleanup).
6. The new admin UI JSON contract is documented in commit messages so
   operators can re-save their backend choice after deploy.

## Out of scope

- Renaming the canonical *string values* (`"huggingface"`, `"openai"`,
  `"gemini"`, `"llama_cpp"`) emitted on OTel attributes. Dashboards keep
  working unchanged.
- Migrating existing operator `.env` files automatically. Operators must
  open the admin UI once after deploy and re-save their selection.
- Multi-PR sub-splitting (Option A in the scope memo). Operator chose
  Option B.
