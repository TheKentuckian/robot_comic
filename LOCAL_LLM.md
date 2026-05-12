# LOCAL_LLM.md — Local LLM Backend Planning & Tracking

> **Living planning document.** Update decisions and check off tasks as they complete.
> Last updated: 2026-05-12

---

## Current State

- **Active model:** `hermes3:8b-llama3.1-q4_K_M` via Ollama v0.23.2 on port 11434
- **Why changing:** Hermes3 8B rejects system prompts > ~2500 tokens; breaks the full Don
  Rickles persona. Also: Ollama has no MTP/speculative decoding path.
- **Stack direction:** Replace Ollama with **llama.cpp + MTP PR #22673** for native
  speculative decoding. llama-server exposes an OpenAI-compatible `/v1/chat/completions`
  endpoint — same wire format, just a different URL.

---

## Decisions Log

### 1. Platform: Native Windows (matching Session 1 VOICE_CLONE.md Decision #1)

**Decision:** Build llama.cpp as a native Windows process. No WSL2.

**Rationale:** WSL2 adds CUDA passthrough latency, slower VirtioFS model loads, and
breaks the "services survive lock screen" property. Native CUDA on RTX 4080 Mobile is
first-class. Single client (Pi), sequential requests — Ollama's multi-user penalty never
applied, and llama.cpp's single-slot mode is a better fit anyway.

**Build toolchain:** MSVC (Visual Studio Build Tools) + CMake. No GCC cross-compile.
**Service wrapper:** PowerShell script added to `Start-RobotServices.ps1`. No systemd.
**Build location:** `D:\Projects\llama.cpp\` (Dev Drive, NVMe-optimized).

### 2. Model: Qwen3 35B-A3B Q4_K_M (MoE)

**Decision:** Qwen3 MoE over Qwen2.5 14B dense or Qwen3 14B dense.

**Rationale:**
- MoE shape: ~35B total weights, ~3.6B active params per forward pass.
  Compute cost ≈ a 3.6B dense model; quality ≈ a much larger model on
  conversational tasks.
- 3–4K input tokens + short <200-token output + one tool call/turn is ideal
  for MoE inference — low memory bandwidth per token, high throughput.
- Q4_K_M: ~20–22 GB model file. With `--n-cpu-moe` offloading MoE expert
  layers to CPU RAM, attention and non-MoE layers fit in 12 GB VRAM.
- **CUDA 13.2 is known to produce gibberish with Qwen3.** Pin CUDA 12.4
  (already installed for Chatterbox). Do not upgrade.

**Source:** Unsloth GGUF on HuggingFace (or equivalent). Must verify MTP tensors are
present in the file before building the MTP code path (some GGUFs advertise
`nextn_predict_layers` metadata but ship without the actual draft-head weights).

**Exact Ollama pull command:** N/A — no longer using Ollama for this model.
**Download command:** `huggingface-cli download` or direct GGUF URL (confirm at Task 2).

### 3. LLM Server: llama.cpp + MTP PR #22673

**Decision:** llama.cpp with Multi-Token Prediction speculative decoding.

**Rationale:**
- MTP draft heads (bundled in the GGUF) let the server speculatively draft 3
  tokens per step. On short conversational outputs, expect ~1.4–1.5× decode
  speedup vs. non-speculative — not 2×, because schema-constrained tool-call
  JSON drafts worse than free text (acceptance rate ~40–55% expected vs.
  headline ~75%).
- If MTP acceptance rate is below ~40% on our actual workload, fall back to
  running without `--spec-type mtp`. The overhead is not worth it below that.
- PR #22673 is beta / not yet merged to master. Pin the commit hash used at
  build time in this doc.

**Fetched PR branch:** `ggml-org/llama.cpp` PR #22673 — `spec-mtp` branch or equivalent.
**Commit pinned:** `3bdc61fe03299a84ade6f185819738c9a05200e7`
Branch top commits: `3bdc61fe` cont: simplify, `fea55085` rename files, `cfb386c6` fix batch size, `a7813c71` spec: support MTP

### 4. Context Window: 8192 tokens

**Decision:** `-c 8192` (not 32K).

**Rationale:** MTP draft branch consumes additional VRAM proportional to context length.
Our workload is 3–4K input + <200 tokens output. 8K provides comfortable headroom.
Expanding to 16K later is easy; shrinking after OOM is not fun.

### 5. CPU MoE Offload: `--n-cpu-moe`

**Decision:** Start with `--n-cpu-moe 30` and tune downward by watching `nvidia-smi`.

**Rationale:** The 35B MoE model won't fit in 12 GB VRAM with all layers on GPU.
MoE expert layers are the main spill candidate — they're large, used sparsely.
Attention and shared layers stay on GPU. CPU RAM on this laptop is ample for expert
spill; the RTX 4080 Mobile's 12 GB carries the hot path.

### 6. ElevenLabs TTS (Issue #60 — tracked separately)

Chatterbox is **parked** (not removed) while ElevenLabs becomes the primary TTS output.
This frees ~4.5 GB VRAM, which is what makes a 35B MoE model viable in the 12 GB budget.

Implementation tracked in GitHub issue #60. Config keys: `ELEVENLABS_API_KEY`,
`ELEVENLABS_VOICE_ID`. New handler file modeled on `gemini_tts.py`.

ElevenLabs is a cloud API — 0 VRAM overhead.

---

## VRAM Budget (New Stack)

| Process                          | Est. VRAM     |
|----------------------------------|---------------|
| llama.cpp (35B MoE, GPU layers)  | ~7–9 GB       |
| MTP draft head                   | ~0.3–0.5 GB   |
| Windows compositor + Chrome      | ~0.5–1.0 GB   |
| ElevenLabs TTS                   | 0 (cloud API) |
| Chatterbox (parked)              | 0             |
| **Total**                        | **~8–10.5 GB**|

~1.5–4 GB headroom vs. the 12 GB cap. Tune `--n-cpu-moe` to reclaim headroom.

---

## Launch Script (Windows/PowerShell)

```powershell
# llama-server launch — to be added to Start-RobotServices.ps1
# Tune --n-cpu-moe after benchmarking (see Task 4)
$env:CUDA_VISIBLE_DEVICES = "0"  # Single GPU, explicit
& "D:\Projects\llama.cpp\build\bin\Release\llama-server.exe" `
    -m "D:\Projects\models\qwen3-35b-a3b-q4_k_m-mtp.gguf" `
    -c 8192 `
    -ngl 999 `
    --n-cpu-moe 30 `
    -fa on `
    --cache-type-k q8_0 `
    --spec-type mtp `
    --spec-num-draft 3 `
    -t 8 `
    -b 2048 `
    -ub 2048 `
    --jinja `
    --host 0.0.0.0 `
    --port 11434 `
    >> "C:\Logs\llama-server.log" 2>&1
```

Note: Using port 11434 to maintain backward compatibility with existing Pi `.env` files.
llama-server's `/v1/chat/completions` is OpenAI-compatible.

---

## Code Changes Required in robot_comic

### 1. `config.py` — Add `LLAMA_CPP_URL`

The current `_ollama_base_url` property in `chatterbox_tts.py` derives the LLM URL
from `CHATTERBOX_URL` by swapping the port to 11434 — fragile. Add a dedicated var:

```python
LLAMA_CPP_URL_ENV = "LLAMA_CPP_URL"
LLAMA_CPP_DEFAULT_URL = "http://astralplane.lan:11434"
# in Config class:
LLAMA_CPP_URL = os.getenv(LLAMA_CPP_URL_ENV, LLAMA_CPP_DEFAULT_URL)
# also add to LOCAL_STT_RESPONSE_BACKEND_CHOICES:
LLAMA_CPP_OUTPUT = "llama_cpp"
```

### 2. `chatterbox_tts.py` — Switch from `/api/chat` to `/v1/chat/completions`

Current: `POST {ollama_base_url}/api/chat` (Ollama-specific format)
New: `POST {llama_cpp_url}/v1/chat/completions` (OpenAI-compatible)

Response shape changes: `data["message"]` → `data["choices"][0]["message"]`

### 3. Hermes3-Specific Workarounds to Remove/Audit

These exist solely because Hermes3 8B has unreliable tool call formatting:
- `_trim_tool_spec()` — truncates enum lists and descriptions to keep prompt under 2500 tokens. **Remove** once verified Qwen3 handles full specs.
- `_coerce_text_tool_call()` — detects `{function: name, ...}` text-format calls. **Remove** after 20-turn sanity check confirms Qwen3 uses structured tool_calls.
- `_nudge_llm()` — retries with a "please use a tool call now" nudge. **Deprecate** (keep as fallback initially, remove after validation).
- `_TEXT_TOOL_CALL_RE`, `_parse_text_tool_args()` — regex parsing for text-format calls. **Remove** with `_coerce_text_tool_call`.
- `_parse_json_content_tool_call()` — JSON-in-content fallback. **Remove** after validation.

Do **not** remove before the 20-turn sanity check in Task 5.

### 4. Rename handler references

The class and module are named `ChatterboxTTS*` but will be serving llama.cpp LLM + ElevenLabs TTS. Rename in a follow-up (Issue #60 scope).

---

## Task Checklist

### Task 1: Clone llama.cpp and merge MTP PR #22673 ✅

- [x] VS 2022 BuildTools already installed (cmake + ninja bundled)
- [x] Clone: `git clone https://github.com/ggml-org/llama.cpp D:\Projects\llama.cpp`
- [x] Fetch PR branch and checkout `spec-mtp` — commit `3bdc61fe03299a84ade6f185819738c9a05200e7`
- [x] CUDA 12.4 installed via full local installer `cuda_12.4.0_551.61_windows.exe` (cuBLAS separate component required)
- [x] Built with Ninja + MSVC + CUDA 12.4: `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`
- [x] Verified: `llama-server version 9117 (3bdc61fe0)` — RTX 4080 Laptop GPU 12281 MiB sm_8.9 ✓
- **Binary:** `D:\Projects\llama.cpp\build\bin\llama-server.exe`
- **Note:** Run with `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` on PATH (handled in Start-RobotServices.ps1)

### Task 2: Download Qwen3 35B-A3B Q4_K_M GGUF ✅

- [x] Source: `unsloth/Qwen3.6-35B-A3B-GGUF-MTP` on HuggingFace
- [x] File: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — 22.7 GB (21.11 binary GB on disk)
- [x] Downloaded to `D:\Projects\models\` via `curl.exe`
- [x] MTP tensors confirmed: `nextn_predict_layers` metadata key present ✓
- **Drive choice:** D (ReFS Dev Drive) over C (faster Samsung NTFS) — Defender AV bypass closes the speed gap on large cold loads

### Task 3: Wire up Start-RobotServices.ps1 ✅

- [x] llama-server launch added to `C:\Services\scripts\Start-RobotServices.ps1`
- [x] Ollama launch removed; Chatterbox commented out (parked)
- [ ] Update Stop-RobotServices.ps1 to kill `llama-server.exe`
- [ ] Test: stop all services, start via shortcut, confirm llama-server log shows model loaded (requires Task 2 model download)

### Task 4: Benchmark on actual workload ⏸️ SHOW RESULTS BEFORE PROCEEDING

Run with representative Don Rickles system prompt (full, no trimming):
- [ ] Prefill time on a 3.5K-token prompt:
  ```powershell
  # Use llama-bench or curl timing on /v1/chat/completions
  # Record: prefill tok/s, time-to-first-token
  ```
- [ ] Decode tok/s with MTP on vs off:
  ```powershell
  # MTP on:  --spec-type mtp --spec-num-draft 3
  # MTP off: remove those flags, restart server
  ```
- [ ] MTP acceptance rate (check llama-server logs or `/metrics` endpoint):
  - Specifically during the tool-call JSON portion of output
  - If < 40% → fall back to non-MTP and note here
- [ ] Tune `--n-cpu-moe` (start at 30, decrease toward 20, watch `nvidia-smi`):
  ```powershell
  # Target: VRAM usage stable, GPU utilization high, no OOM
  nvidia-smi dmon -s mu -d 2
  ```
- [ ] **Results:**
  ```
  Prefill:     ___  tok/s  |  TTFT: ___ ms
  Decode MTP:  ___  tok/s  |  Acceptance: ___% (tool-call portion: ___%  )
  Decode base: ___  tok/s  |  Speedup: ___×
  VRAM used:   ___ GB      |  --n-cpu-moe final: ___
  ```

### Task 5: Sanity-check tool-call JSON well-formedness

- [ ] Run 20 representative turns with MTP enabled, greedy decoding (`temperature=0`)
- [ ] Verify every turn produces valid JSON tool_calls (no text-format fallback needed)
- [ ] Compare output vs. non-MTP run — should be byte-identical (greedy = deterministic)
- [ ] If Qwen3 is clean: remove Hermes3 workarounds from `chatterbox_tts.py` (see Code Changes §3)

### Task 6: Update config.py and chatterbox_tts.py

- [ ] Add `LLAMA_CPP_URL` to `config.py` (see Code Changes §1)
- [ ] Update `_call_llm()` in `chatterbox_tts.py` to use `/v1/chat/completions` (see Code Changes §2)
- [ ] Update `_ollama_base_url` property → `_llama_cpp_url` reading new config var
- [ ] Remove Hermes3 workarounds (after Task 5 confirms clean behavior)
- [ ] Run existing test suite: `pytest tests/ -v`
- [ ] Update Pi `.env`: `LLAMA_CPP_URL=http://astralplane.lan:11434`

### Task 7: Update OLLAMA_MODEL references

- [ ] `config.py`: rename `OLLAMA_MODEL_DEFAULT` → keep for backward compat or remove
- [ ] `config.py`: add `LLAMA_CPP_MODEL` env var (optional, for logging/telemetry only)
- [ ] Update Issue #46 (validate OLLAMA_MODEL at startup) — now validates llama-server instead
- [ ] Commit + push

---

## Open Questions

- [ ] Exact Qwen3 35B-A3B GGUF name and Unsloth repo URL? *(confirm at Task 2)*
- [ ] Does PR #22673 build cleanly on MSVC, or does it require a MinGW/Clang workaround?
- [ ] What is the actual `--n-cpu-moe` sweet spot for this VRAM budget? *(fill in at Task 4)*
- [ ] MTP acceptance rate on tool-call JSON specifically? *(fill in at Task 4)*
- [ ] Is Qwen3's `--jinja` Jinja2 chat template compatible with our tool schema format? *(confirm at Task 5)*

---

## Reference Links

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [PR #22673 — MTP speculative decoding](https://github.com/ggml-org/llama.cpp/pull/22673)
- [Unsloth Qwen3 GGUFs on HuggingFace](https://huggingface.co/unsloth)
- [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)
- [CUDA 12.4 PyTorch wheels](https://download.pytorch.org/whl/cu124) — keep pinned, do NOT upgrade to 13.2
- GitHub Issue #59 — Switch local LLM from Hermes3 8B to Qwen
- GitHub Issue #60 — ElevenLabs TTS integration (tracks TTS side of this work)
