const OPENAI_BACKEND = "openai";
const GEMINI_BACKEND = "gemini";
const HF_BACKEND = "huggingface";
const LOCAL_STT_BACKEND = "local_stt";
const GEMINI_TTS_OUTPUT = "gemini_tts";
const CHATTERBOX_OUTPUT = "chatterbox";
const ELEVENLABS_OUTPUT = "elevenlabs";
const LLAMA_ELEVENLABS_TTS_OUTPUT = "llama_elevenlabs_tts";
const DEFAULT_BACKEND = HF_BACKEND;
const HF_DEFAULT_HOST = "localhost";
const HF_DEFAULT_PORT = 8765;
const BACKEND_META = {
  [OPENAI_BACKEND]: {
    label: "OpenAI Realtime",
    formTitle: "Connect OpenAI",
    inputLabel: "OpenAI API Key",
    placeholder: "sk-...",
    saveButton: "Save key",
    changeButton: "Change OpenAI key",
    readyTitle: "OpenAI Realtime ready",
    readyCopy: "OpenAI Realtime is configured. Your saved OpenAI key is ready to use.",
    formCopy: "Paste your OPENAI_API_KEY once and we will store it locally for the headless conversation loop.",
    requiredCredentialsCopy: "OpenAI Realtime requires your own OPENAI_API_KEY before you can switch.",
    note: "OpenAI Realtime requires your own OPENAI_API_KEY.",
  },
  [GEMINI_BACKEND]: {
    label: "Gemini Live",
    formTitle: "Connect Gemini Live",
    inputLabel: "GEMINI_API_KEY",
    placeholder: "AIza...",
    saveButton: "Save token",
    changeButton: "Change Gemini token",
    readyTitle: "Gemini Live ready",
    readyCopy: "Gemini Live is configured. Your saved Gemini token is ready to use.",
    formCopy: "Paste your GEMINI_API_KEY once and we will store it locally for the headless conversation loop.",
    requiredCredentialsCopy: "Gemini Live requires your own GEMINI_API_KEY before you can switch.",
    note: "OpenAI Realtime requires OPENAI_API_KEY. Gemini Live needs GEMINI_API_KEY.",
  },
  [HF_BACKEND]: {
    label: "Hugging Face",
    formTitle: "Configure Hugging Face",
    inputLabel: "",
    placeholder: "",
    saveButton: "Save connection",
    changeButton: "Edit connection",
    readyTitle: "Hugging Face ready",
    readyCopy: "Hugging Face is configured. You can jump straight to personalities.",
    formCopy: "Choose where Reachy should connect for Hugging Face.",
    requiredCredentialsCopy: "Set up the Hugging Face connection details before switching.",
    note: "Hugging Face can use the built-in server or your own local realtime websocket.",
  },
  [LOCAL_STT_BACKEND]: {
    label: "Local STT",
    formTitle: "Configure Local STT",
    inputLabel: "OPENAI_API_KEY",
    placeholder: "sk-...",
    saveButton: "Save local STT",
    changeButton: "Edit local STT",
    readyTitle: "Local STT ready",
    readyCopy: "Moonshine will transcribe speech on-device, then the selected output backend will generate the spoken response.",
    formCopy: "Choose a local Moonshine model and a separate voice output backend.",
    requiredCredentialsCopy: "Local STT needs credentials or connection details for the selected output backend.",
    note: "Local STT keeps speech recognition on-device, then sends text to the selected voice output backend.",
  },
};
const JOURNEY_META = {
  [HF_BACKEND]: {
    inputLabel: "Hugging Face realtime audio",
    inputCopy: "Speech streams to the Hugging Face voice backend.",
    brainLabel: "Hugging Face response model",
    brainCopy: "The configured endpoint handles speech understanding, tools, and replies.",
    outputLabel: "Hugging Face voice",
    outputCopy: "Audio returns from the built-in server or your direct endpoint.",
  },
  [OPENAI_BACKEND]: {
    inputLabel: "OpenAI Realtime audio",
    inputCopy: "Microphone audio streams directly to OpenAI.",
    brainLabel: "OpenAI Realtime",
    brainCopy: "Realtime handles understanding, personality, tools, and response timing.",
    outputLabel: "OpenAI voice",
    outputCopy: "Speech comes back through the selected OpenAI voice.",
  },
  [GEMINI_BACKEND]: {
    inputLabel: "Gemini Live audio",
    inputCopy: "Microphone audio streams directly to Gemini Live.",
    brainLabel: "Gemini Live",
    brainCopy: "Gemini handles understanding, personality, tools, and response timing.",
    outputLabel: "Gemini Live voice",
    outputCopy: "Speech comes back through the selected Gemini voice.",
  },
  [LOCAL_STT_BACKEND]: {
    inputLabel: "Moonshine STT",
    inputCopy: "Speech recognition runs locally on the robot.",
    brainLabel: "Text response backend",
    brainCopy: "Robot Comic sends transcripts to the selected output backend.",
    outputLabel: "OpenAI voice",
    outputCopy: "Choose OpenAI or Hugging Face today; Gemini Flash 3.1 TTS is reserved as a future route.",
  },
};

function backendHasCredentials(status, backend) {
  if (backend === GEMINI_BACKEND) return !!status.has_gemini_key;
  if (backend === HF_BACKEND) return !!(status.has_hf_connection ?? (status.has_hf_session_url || status.has_hf_ws_url));
  if (backend === LOCAL_STT_BACKEND) return !!status.has_local_stt_key;
  return !!status.has_openai_key;
}

function backendCanProceed(status, backend) {
  if (backend === GEMINI_BACKEND) {
    return status.can_proceed_with_gemini !== undefined
      ? !!status.can_proceed_with_gemini
      : backendHasCredentials(status, backend);
  }
  if (backend === HF_BACKEND) {
    return status.can_proceed_with_hf !== undefined
      ? !!status.can_proceed_with_hf
      : backendHasCredentials(status, backend);
  }
  if (backend === LOCAL_STT_BACKEND) {
    return status.can_proceed_with_local_stt !== undefined
      ? !!status.can_proceed_with_local_stt
      : backendHasCredentials(status, backend);
  }
  return status.can_proceed_with_openai !== undefined
    ? !!status.can_proceed_with_openai
    : backendHasCredentials(status, backend);
}

function backendMeta(backend) {
  return BACKEND_META[backend] || BACKEND_META[DEFAULT_BACKEND];
}

function journeyMeta(backend, outputBackend = OPENAI_BACKEND) {
  const meta = { ...(JOURNEY_META[backend] || JOURNEY_META[DEFAULT_BACKEND]) };
  if (backend === LOCAL_STT_BACKEND) {
    if (outputBackend === HF_BACKEND) {
      meta.brainLabel = "Hugging Face response backend";
      meta.outputLabel = "Hugging Face voice";
      meta.outputCopy = "Speech comes back through the configured Hugging Face endpoint.";
    } else if (outputBackend === GEMINI_TTS_OUTPUT) {
      meta.brainLabel = "Gemini Flash response backend";
      meta.outputLabel = "Gemini Flash 3.1 TTS";
      meta.outputCopy = "Speech comes back through Gemini 3.1 Flash TTS with the Algenib voice.";
    } else if (outputBackend === CHATTERBOX_OUTPUT) {
      meta.brainLabel = "Ollama LLM (local)";
      meta.outputLabel = "Chatterbox TTS";
      meta.outputCopy = "Text goes to the local Ollama model, then to Chatterbox voice-clone TTS.";
    } else if (outputBackend === ELEVENLABS_OUTPUT) {
      meta.brainLabel = "Gemini Flash response backend";
      meta.outputLabel = "ElevenLabs TTS";
      meta.outputCopy = "Text goes to Gemini Flash, then to the selected ElevenLabs voice.";
    } else if (outputBackend === LLAMA_ELEVENLABS_TTS_OUTPUT) {
      meta.brainLabel = "llama.cpp (local LLM)";
      meta.outputLabel = "ElevenLabs TTS";
      meta.outputCopy = "Text goes to the local llama.cpp server, then to the selected ElevenLabs voice.";
    } else {
      meta.brainLabel = "OpenAI response backend";
      meta.outputLabel = "OpenAI voice";
      meta.outputCopy = "Speech comes back through OpenAI text-in, audio-out realtime.";
    }
  }
  return meta;
}

function formatBackendNote(text) {
  return text
    .replace("GEMINI_API_KEY", "<code>GEMINI_API_KEY</code>")
    .replace("HF_REALTIME_WS_URL", "<code>HF_REALTIME_WS_URL</code>");
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function waitForStatus(timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

async function waitForPersonalityData(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  let attempts = 0;
  const deadline = Date.now() + timeoutMs;
  while (true) {
    attempts += 1;
    try {
      const url = new URL("/personalities", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}

    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend…" : "Loading…";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

async function validateKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/validate_api_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "validation_failed");
  }
  return data;
}

async function saveBackendConfig(backend, { key = "", hfMode = "", hfHost = "", hfPort = null } = {}) {
  const body = { backend, api_key: key };
  if (backend === HF_BACKEND) {
    if (hfMode) body.hf_mode = hfMode;
    if (hfHost) body.hf_host = hfHost;
    if (hfPort !== null && hfPort !== undefined) body.hf_port = hfPort;
  }
  if (backend === LOCAL_STT_BACKEND) {
    const languageEl = document.getElementById("local-stt-language");
    const cacheEl = document.getElementById("local-stt-cache");
    const responseEl = document.getElementById("local-stt-response");
    const modelEl = document.getElementById("local-stt-model");
    const updateEl = document.getElementById("local-stt-update");
    body.local_stt_language = (languageEl?.value || "en").trim();
    body.local_stt_cache_dir = (cacheEl?.value || "./cache/moonshine_voice").trim();
    body.local_stt_response_backend = responseEl?.value || OPENAI_BACKEND;
    body.local_stt_model = modelEl?.value || "tiny_streaming";
    const updateInterval = Number.parseFloat((updateEl?.value || "0.35").trim());
    if (Number.isFinite(updateInterval)) body.local_stt_update_interval = updateInterval;
    if (responseEl?.value === CHATTERBOX_OUTPUT) {
      const urlEl = document.getElementById("chatterbox-url");
      const voiceEl = document.getElementById("chatterbox-voice");
      if (urlEl?.value.trim()) body.chatterbox_url = urlEl.value.trim();
      if (voiceEl?.value.trim()) body.chatterbox_voice = voiceEl.value.trim();
    }
    if (
      responseEl?.value === ELEVENLABS_OUTPUT
      || responseEl?.value === LLAMA_ELEVENLABS_TTS_OUTPUT
    ) {
      const keyEl = document.getElementById("elevenlabs-key");
      const voiceEl = document.getElementById("elevenlabs-voice");
      if (keyEl?.value.trim()) body.elevenlabs_api_key = keyEl.value.trim();
      if (voiceEl?.value.trim()) body.elevenlabs_voice = voiceEl.value.trim();
    }
  }
  const resp = await fetch("/backend_config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

// ---------- Personalities API ----------
async function loadPersonality(name) {
  const url = new URL("/personalities/load", window.location.origin);
  url.searchParams.set("name", name);
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, {}, 3000);
  if (!resp.ok) throw new Error("load_failed");
  return await resp.json();
}

async function savePersonality(payload) {
  // Try JSON POST first
  const saveUrl = new URL("/personalities/save", window.location.origin);
  saveUrl.searchParams.set("_", Date.now().toString());
  let resp = await fetchWithTimeout(saveUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, 5000);
  if (resp.ok) return await resp.json();

  // Fallback to form-encoded POST
  try {
    const form = new URLSearchParams();
    form.set("name", payload.name || "");
    form.set("instructions", payload.instructions || "");
    form.set("tools_text", payload.tools_text || "");
    form.set("voice", payload.voice || "");
    const url = new URL("/personalities/save_raw", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    resp = await fetchWithTimeout(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    }, 5000);
    if (resp.ok) return await resp.json();
  } catch {}

  // Fallback to GET (query params)
  try {
    const url = new URL("/personalities/save_raw", window.location.origin);
    url.searchParams.set("name", payload.name || "");
    url.searchParams.set("instructions", payload.instructions || "");
    url.searchParams.set("tools_text", payload.tools_text || "");
    url.searchParams.set("voice", payload.voice || "");
    url.searchParams.set("_", Date.now().toString());
    resp = await fetchWithTimeout(url, { method: "GET" }, 5000);
    if (resp.ok) return await resp.json();
  } catch {}

  const data = await resp.json().catch(() => ({}));
  throw new Error(data.error || "save_failed");
}

async function applyVoice(voice) {
  const url = new URL("/voices/apply", window.location.origin);
  url.searchParams.set("voice", voice || "");
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, { method: "POST" }, 5000);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "apply_voice_failed");
  }
  return await resp.json();
}

async function applyPersonality(name, { persist = false } = {}) {
  // Send as query param to avoid any body parsing issues on the server
  const url = new URL("/personalities/apply", window.location.origin);
  url.searchParams.set("name", name || "");
  if (persist) {
    url.searchParams.set("persist", "1");
  }
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, { method: "POST" }, 5000);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "apply_failed");
  }
  return await resp.json();
}

async function clearCrowdHistory() {
  const resp = await fetchWithTimeout("/crowd_history/clear", { method: "POST" }, 5000);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "clear_failed");
  }
  return data;
}

async function getPausePhrases() {
  try {
    const url = new URL("/pause_phrases", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("pause_phrases_failed");
    return await resp.json();
  } catch (e) {
    return null;
  }
}

function parsePhraseTextarea(value) {
  if (typeof value !== "string") return [];
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function savePausePhrases({ stop, resume, shutdown, switch: switchPhrases }) {
  const resp = await fetchWithTimeout(
    "/pause_phrases",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stop: stop.length ? stop : null,
        resume: resume.length ? resume : null,
        shutdown: shutdown.length ? shutdown : null,
        switch: switchPhrases.length ? switchPhrases : null,
      }),
    },
    5000,
  );
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "save_failed");
  }
  return data;
}

async function restartApp() {
  const resp = await fetchWithTimeout("/admin/restart", { method: "POST" }, 5000);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "restart_failed");
  }
  return data;
}

async function getMovementSpeed() {
  try {
    const url = new URL("/movement_speed", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function setMovementSpeed(value) {
  const resp = await fetchWithTimeout(
    "/movement_speed",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    },
    3000,
  );
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "set_failed");
  }
  return data;
}

async function getVoices() {
  try {
    const url = new URL("/voices", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("voices_failed");
    return await resp.json();
  } catch (e) {
    return [];
  }
}

async function getCurrentVoice() {
  try {
    const url = new URL("/voices/current", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("current_voice_failed");
    const data = await resp.json();
    return typeof data.voice === "string" ? data.voice : "";
  } catch (e) {
    return "";
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

function setStatusMessage(el, text, tone = "") {
  el.textContent = text;
  el.className = tone ? `status ${tone}` : "status";
  el.setAttribute("role", tone === "error" ? "alert" : "status");
  el.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
  el.setAttribute("aria-atomic", "true");
}

function describeHFConfiguration(status) {
  if (status.hf_connection_mode === "local") {
    const host = status.hf_direct_host || HF_DEFAULT_HOST;
    const port = status.hf_direct_port || HF_DEFAULT_PORT;
    return `Hugging Face will connect directly to ${host}:${port}.`;
  }
  if (status.has_hf_session_url) {
    return "Hugging Face will use the built-in server.";
  }
  return "Choose the Hugging Face server or a local realtime endpoint.";
}

function isLocalHFHost(host) {
  return !host || host === "localhost" || host === "127.0.0.1";
}

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);
  const backendChip = document.getElementById("backend-chip");
  const backendNote = document.getElementById("backend-note");
  const backendStatusEl = document.getElementById("backend-status");
  const backendSaveBtn = document.getElementById("save-backend-btn");
  const backendInputs = Array.from(document.querySelectorAll('input[name="backend"]'));
  const backendCards = Array.from(document.querySelectorAll("[data-backend-card]"));
  const journeyInputLabel = document.getElementById("journey-input-label");
  const journeyInputCopy = document.getElementById("journey-input-copy");
  const journeyBrainLabel = document.getElementById("journey-brain-label");
  const journeyBrainCopy = document.getElementById("journey-brain-copy");
  const journeyOutputLabel = document.getElementById("journey-output-label");
  const journeyOutputCopy = document.getElementById("journey-output-copy");
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const configuredTitle = document.getElementById("configured-title");
  const configuredCopy = document.getElementById("configured-copy");
  const crowdHistoryChip = document.getElementById("crowd-history-chip");
  const crowdHistoryPath = document.getElementById("crowd-history-path");
  const crowdHistoryStatus = document.getElementById("crowd-history-status");
  const clearCrowdHistoryBtn = document.getElementById("clear-crowd-history");
  const personalityPanel = document.getElementById("personality-panel");
  const formTitle = document.getElementById("form-title");
  const formCopy = document.getElementById("form-copy");
  const apiKeyFields = document.getElementById("api-key-fields");
  const apiKeyLabel = document.getElementById("api-key-label");
  const saveBtn = document.getElementById("save-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const input = document.getElementById("api-key");
  const hfFields = document.getElementById("hf-fields");
  const hfMode = document.getElementById("hf-mode");
  const hfDirectFields = document.getElementById("hf-direct-fields");
  const hfHostPreset = document.getElementById("hf-host-preset");
  const hfHostCustomWrap = document.getElementById("hf-host-custom-wrap");
  const hfHostCustom = document.getElementById("hf-host-custom");
  const hfPort = document.getElementById("hf-port");
  const hfPreview = document.getElementById("hf-preview");
  const localSttFields = document.getElementById("local-stt-fields");
  const localSttLanguage = document.getElementById("local-stt-language");
  const localSttCache = document.getElementById("local-stt-cache");
  const localSttResponse = document.getElementById("local-stt-response");
  const localSttModel = document.getElementById("local-stt-model");
  const localSttUpdate = document.getElementById("local-stt-update");
  const localSttOutputInputs = Array.from(document.querySelectorAll('input[name="local-stt-output"]'));
  const localSttOutputCards = Array.from(document.querySelectorAll("[data-output-card]"));

  // Personality elements
  const pSelect = document.getElementById("personality-select");
  const pApply = document.getElementById("apply-personality");
  const pPersist = document.getElementById("persist-personality");
  const pNew = document.getElementById("new-personality");
  const pSave = document.getElementById("save-personality");
  const pStartupLabel = document.getElementById("startup-label");
  const pName = document.getElementById("personality-name");
  const pInstr = document.getElementById("instructions-ta");
  const pTools = document.getElementById("tools-ta");
  const pStatus = document.getElementById("personality-status");
  const pVoice = document.getElementById("voice-select");
  const pApplyVoice = document.getElementById("apply-voice");
  const pAvail = document.getElementById("tools-available");

  const AUTO_WITH = {
    dance: ["stop_dance"],
    play_emotion: ["stop_emotion"],
  };
  let selectedBackend = DEFAULT_BACKEND;
  let editingCredentials = false;

  function resolveHFHost() {
    return hfHostPreset.value === "custom" ? hfHostCustom.value.trim() : HF_DEFAULT_HOST;
  }

  function updateHFControls() {
    const localMode = hfMode.value !== "deployed";
    const customHost = hfHostPreset.value === "custom";
    show(hfDirectFields, localMode);
    show(hfHostCustomWrap, localMode && customHost);

    if (!localMode) {
      setStatusMessage(hfPreview, "Hugging Face will use the built-in server.");
      return;
    }

    const host = resolveHFHost() || "<host>";
    const port = (hfPort.value || String(HF_DEFAULT_PORT)).trim();
    setStatusMessage(hfPreview, `Will save ws://${host}:${port}/v1/realtime`);
  }

  function populateHFFields(status) {
    const mode = status.hf_connection_mode
      || (status.has_hf_session_url ? "deployed" : "local");
    const existingHost = status.hf_direct_host || HF_DEFAULT_HOST;
    const existingPort = status.hf_direct_port || HF_DEFAULT_PORT;

    hfMode.value = mode;
    if (isLocalHFHost(existingHost)) {
      hfHostPreset.value = "localhost";
      hfHostCustom.value = "";
    } else {
      hfHostPreset.value = "custom";
      hfHostCustom.value = existingHost;
    }
    hfPort.value = String(existingPort);
    updateHFControls();
  }

  function populateLocalSTTFields(status) {
    localSttLanguage.value = status.local_stt_language || "en";
    localSttCache.value = status.local_stt_cache_dir || "./cache/moonshine_voice";
    localSttResponse.value = status.local_stt_response_backend || OPENAI_BACKEND;
    localSttModel.innerHTML = "";
    const choices = Array.isArray(status.local_stt_model_choices) && status.local_stt_model_choices.length
      ? status.local_stt_model_choices
      : ["tiny_streaming", "small_streaming"];
    for (const choice of choices) {
      const opt = document.createElement("option");
      opt.value = choice;
      opt.textContent = choice === "small_streaming" ? "Small streaming" : "Tiny streaming";
      localSttModel.appendChild(opt);
    }
    localSttModel.value = choices.includes(status.local_stt_model) ? status.local_stt_model : choices[0];
    localSttUpdate.value = String(status.local_stt_update_interval || 0.35);
    const chatterboxUrl = document.getElementById("chatterbox-url");
    const chatterboxVoice = document.getElementById("chatterbox-voice");
    if (chatterboxUrl) chatterboxUrl.value = status.chatterbox_url || "http://astralplane.lan:8004";
    if (chatterboxVoice) chatterboxVoice.value = status.chatterbox_voice || "don_rickles";
    elevenlabsSavedVoice = status.elevenlabs_voice || "";
    setSelectedLocalSTTOutput(localSttResponse.value || OPENAI_BACKEND);
  }

  let elevenlabsVoicesLoaded = false;
  let elevenlabsSavedVoice = "";
  async function populateElevenLabsVoices() {
    const select = document.getElementById("elevenlabs-voice");
    if (!select) return;
    if (elevenlabsVoicesLoaded) {
      if (elevenlabsSavedVoice) {
        const match = Array.from(select.options).find((o) => o.value === elevenlabsSavedVoice);
        if (match) select.value = elevenlabsSavedVoice;
      }
      return;
    }
    elevenlabsVoicesLoaded = true;
    select.innerHTML = "";
    const loading = document.createElement("option");
    loading.value = "";
    loading.textContent = "(loading voices…)";
    select.appendChild(loading);
    try {
      const url = new URL("/elevenlabs/voices", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 5000);
      if (!resp.ok) throw new Error("voices_failed");
      const data = await resp.json();
      const voices = Array.isArray(data?.voices) ? data.voices : [];
      select.innerHTML = "";
      if (!voices.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "(no voices available)";
        select.appendChild(opt);
        return;
      }
      for (const v of voices) {
        const opt = document.createElement("option");
        const name = (v && typeof v.name === "string") ? v.name : "";
        if (!name) continue;
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
      if (elevenlabsSavedVoice) {
        const match = Array.from(select.options).find((o) => o.value === elevenlabsSavedVoice);
        if (match) select.value = elevenlabsSavedVoice;
      }
    } catch (e) {
      select.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(unable to load voices)";
      select.appendChild(opt);
      // Allow retry on next selection click.
      elevenlabsVoicesLoaded = false;
    }
  }

  function renderCrowdHistory(status) {
    const count = Number.parseInt(status.crowd_history_count ?? 0, 10) || 0;
    crowdHistoryChip.textContent = `${count} stored`;
    crowdHistoryPath.textContent = status.crowd_history_dir
      ? `Local path: ${status.crowd_history_dir}`
      : "Local path unavailable.";
    clearCrowdHistoryBtn.disabled = count === 0;
  }

  function setSelectedBackend(backend) {
    selectedBackend = [OPENAI_BACKEND, GEMINI_BACKEND, HF_BACKEND, LOCAL_STT_BACKEND].includes(backend)
      ? backend
      : DEFAULT_BACKEND;
    backendInputs.forEach((radio) => {
      radio.checked = radio.value === selectedBackend;
    });
    backendCards.forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.backendCard === selectedBackend);
    });
  }

  function setSelectedLocalSTTOutput(outputBackend) {
    const normalized = outputBackend === HF_BACKEND
      ? HF_BACKEND
      : outputBackend === GEMINI_TTS_OUTPUT
        ? GEMINI_TTS_OUTPUT
        : outputBackend === CHATTERBOX_OUTPUT
          ? CHATTERBOX_OUTPUT
          : outputBackend === ELEVENLABS_OUTPUT
            ? ELEVENLABS_OUTPUT
            : outputBackend === LLAMA_ELEVENLABS_TTS_OUTPUT
              ? LLAMA_ELEVENLABS_TTS_OUTPUT
              : OPENAI_BACKEND;
    localSttResponse.value = normalized;
    localSttOutputInputs.forEach((radio) => {
      radio.checked = radio.value === normalized;
    });
    localSttOutputCards.forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.outputCard === normalized);
    });
    const chatterboxFields = document.getElementById("chatterbox-fields");
    if (chatterboxFields) chatterboxFields.style.display = normalized === CHATTERBOX_OUTPUT ? "" : "none";
    const elevenlabsFields = document.getElementById("elevenlabs-fields");
    const usesElevenLabs = normalized === ELEVENLABS_OUTPUT || normalized === LLAMA_ELEVENLABS_TTS_OUTPUT;
    if (elevenlabsFields) elevenlabsFields.style.display = usesElevenLabs ? "" : "none";
    if (usesElevenLabs) {
      // Fetch voices lazily; safe to call repeatedly.
      populateElevenLabsVoices();
    }
  }

  function renderJourneyMap() {
    const meta = journeyMeta(selectedBackend, localSttResponse.value || OPENAI_BACKEND);
    journeyInputLabel.textContent = meta.inputLabel;
    journeyInputCopy.textContent = meta.inputCopy;
    journeyBrainLabel.textContent = meta.brainLabel;
    journeyBrainCopy.textContent = meta.brainCopy;
    journeyOutputLabel.textContent = meta.outputLabel;
    journeyOutputCopy.textContent = meta.outputCopy;
  }

  function renderCredentialPanels(status) {
    const persistedBackend = status.backend_provider || DEFAULT_BACKEND;
    const activeBackend = status.active_backend || persistedBackend;
    const requiresRestart = !!status.requires_restart;
    const meta = backendMeta(selectedBackend);
    const selectedMatchesPersisted = selectedBackend === persistedBackend;
    const selectedMatchesActive = selectedBackend === activeBackend;
    const localSttUsesHF = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === HF_BACKEND;
    const localSttUsesGeminiTTS = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === GEMINI_TTS_OUTPUT;
    const localSttUsesChatterbox = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === CHATTERBOX_OUTPUT;
    const localSttUsesElevenLabs = selectedBackend === LOCAL_STT_BACKEND && (
      localSttResponse.value === ELEVENLABS_OUTPUT
      || localSttResponse.value === LLAMA_ELEVENLABS_TTS_OUTPUT
    );
    const localSttUsesOpenAI = selectedBackend === LOCAL_STT_BACKEND && !localSttUsesHF && !localSttUsesGeminiTTS && !localSttUsesChatterbox && !localSttUsesElevenLabs;
    const canProceedWithSelectedBackend = localSttUsesHF
      ? backendCanProceed(status, HF_BACKEND)
      : localSttUsesGeminiTTS
        ? backendCanProceed(status, GEMINI_BACKEND)
        : localSttUsesChatterbox
          ? !!(status.can_proceed_with_chatterbox ?? true)
          : localSttUsesElevenLabs
            ? !!status.has_elevenlabs_key
            : localSttUsesOpenAI
              ? backendCanProceed(status, OPENAI_BACKEND)
              : backendCanProceed(status, selectedBackend);
    const usesApiKeyForm = selectedBackend === OPENAI_BACKEND || selectedBackend === GEMINI_BACKEND || localSttUsesOpenAI || localSttUsesGeminiTTS;
    const usesHFForm = selectedBackend === HF_BACKEND || localSttUsesHF;
    const usesLocalSTTForm = selectedBackend === LOCAL_STT_BACKEND;
    const supportsForm = usesApiKeyForm || usesHFForm || usesLocalSTTForm;

    renderJourneyMap();
    backendChip.textContent = selectedBackend === persistedBackend ? "Saved" : "Selected";
    backendNote.innerHTML = formatBackendNote(meta.note);

    configuredTitle.textContent = meta.readyTitle;
    configuredCopy.textContent = usesHFForm && selectedBackend !== LOCAL_STT_BACKEND
      ? describeHFConfiguration(status)
      : meta.readyCopy;
    formTitle.textContent = meta.formTitle;
    formCopy.textContent = usesHFForm && selectedBackend !== LOCAL_STT_BACKEND
      ? meta.formCopy
      : canProceedWithSelectedBackend
        ? meta.formCopy
        : meta.requiredCredentialsCopy;
    apiKeyLabel.textContent = localSttUsesGeminiTTS ? "GEMINI_API_KEY" : meta.inputLabel;
    input.placeholder = localSttUsesGeminiTTS ? "AIza..." : meta.placeholder;
    saveBtn.textContent = meta.saveButton;
    changeKeyBtn.textContent = meta.changeButton;

    show(configuredPanel, canProceedWithSelectedBackend && !editingCredentials);
    show(formPanel, supportsForm && (editingCredentials || !canProceedWithSelectedBackend));
    show(apiKeyFields, usesApiKeyForm);
    show(localSttFields, usesLocalSTTForm);
    show(hfFields, usesHFForm);
    if (usesHFForm) updateHFControls();
    show(changeKeyBtn, supportsForm && canProceedWithSelectedBackend && !editingCredentials);
    show(
      backendSaveBtn,
      canProceedWithSelectedBackend && !selectedMatchesPersisted && !editingCredentials,
    );
    backendSaveBtn.textContent = `Use ${meta.label}`;

    if (requiresRestart && selectedMatchesPersisted) {
      setStatusMessage(
        backendStatusEl,
        `Backend saved. Restart Robot Comic from the dashboard or desktop app to use ${backendMeta(persistedBackend).label}.`,
        "warn",
      );
    } else if (!selectedMatchesPersisted) {
      setStatusMessage(
        backendStatusEl,
        canProceedWithSelectedBackend
          ? selectedMatchesActive && requiresRestart
            ? `Use ${meta.label} to cancel the pending backend change.`
            : `Ready to switch to ${meta.label}.`
          : meta.requiredCredentialsCopy,
        canProceedWithSelectedBackend ? "" : "warn",
      );
    } else {
      setStatusMessage(backendStatusEl, "");
    }
  }

  statusEl.textContent = "Checking configuration...";
  show(formPanel, false);
  show(configuredPanel, false);
  show(personalityPanel, false);

  const st = (await waitForStatus()) || {
    active_backend: DEFAULT_BACKEND,
    backend_provider: DEFAULT_BACKEND,
    has_key: false,
    has_openai_key: false,
    has_gemini_key: false,
    has_hf_session_url: false,
    has_hf_ws_url: false,
    has_hf_connection: false,
    hf_connection_mode: "local",
    hf_direct_host: HF_DEFAULT_HOST,
    hf_direct_port: HF_DEFAULT_PORT,
    can_proceed: false,
    can_proceed_with_openai: false,
    can_proceed_with_gemini: false,
    can_proceed_with_hf: false,
    has_local_stt_key: false,
    can_proceed_with_local_stt: false,
    local_stt_language: "en",
    local_stt_cache_dir: "./cache/moonshine_voice",
    local_stt_response_backend: OPENAI_BACKEND,
    local_stt_response_backend_choices: [OPENAI_BACKEND, HF_BACKEND],
    local_stt_model: "tiny_streaming",
    local_stt_update_interval: 0.35,
    local_stt_model_choices: ["tiny_streaming", "small_streaming"],
    requires_restart: false,
    crowd_history_dir: "",
    crowd_history_count: 0,
    crowd_history_latest: null,
  };
  populateHFFields(st);
  populateLocalSTTFields(st);
  renderCrowdHistory(st);
  setSelectedBackend(st.backend_provider || DEFAULT_BACKEND);
  statusEl.textContent = "";
  renderCredentialPanels(st);

  clearCrowdHistoryBtn.addEventListener("click", async () => {
    setStatusMessage(crowdHistoryStatus, "Clearing crowd history...");
    clearCrowdHistoryBtn.disabled = true;
    try {
      const data = await clearCrowdHistory();
      renderCrowdHistory(data);
      const plural = data.removed === 1 ? "" : "s";
      setStatusMessage(crowdHistoryStatus, `Cleared ${data.removed || 0} session file${plural}.`, "ok");
    } catch (e) {
      clearCrowdHistoryBtn.disabled = false;
      setStatusMessage(crowdHistoryStatus, "Failed to clear crowd history.", "error");
    }
  });

  // Pause phrases admin section
  const pauseStopEl = document.getElementById("pause-phrase-stop");
  const pauseResumeEl = document.getElementById("pause-phrase-resume");
  const pauseShutdownEl = document.getElementById("pause-phrase-shutdown");
  const pauseSwitchEl = document.getElementById("pause-phrase-switch");
  const pausePhrasesStatus = document.getElementById("pause-phrases-status");
  const savePausePhrasesBtn = document.getElementById("save-pause-phrases");
  const resetPausePhrasesBtn = document.getElementById("reset-pause-phrases");

  function fillPauseField(textarea, savedList, effectiveList) {
    if (!textarea) return;
    if (Array.isArray(savedList) && savedList.length > 0) {
      textarea.value = savedList.join("\n");
      textarea.placeholder = (effectiveList || []).join(", ");
    } else {
      textarea.value = "";
      textarea.placeholder = (effectiveList || []).join(", ");
    }
  }

  function applyPausePhrasePayload(payload) {
    if (!payload) return;
    const saved = payload.saved || {};
    const effective = payload.effective || {};
    fillPauseField(pauseStopEl, saved.stop, effective.stop);
    fillPauseField(pauseResumeEl, saved.resume, effective.resume);
    fillPauseField(pauseShutdownEl, saved.shutdown, effective.shutdown);
    fillPauseField(pauseSwitchEl, saved.switch, effective.switch);
  }

  (async () => {
    const data = await getPausePhrases();
    if (data && data.ok) {
      applyPausePhrasePayload(data);
    } else {
      setStatusMessage(pausePhrasesStatus, "Could not load saved phrases.", "error");
    }
  })();

  if (savePausePhrasesBtn) {
    savePausePhrasesBtn.addEventListener("click", async () => {
      setStatusMessage(pausePhrasesStatus, "Saving phrases…");
      savePausePhrasesBtn.disabled = true;
      try {
        const data = await savePausePhrases({
          stop: parsePhraseTextarea(pauseStopEl.value),
          resume: parsePhraseTextarea(pauseResumeEl.value),
          shutdown: parsePhraseTextarea(pauseShutdownEl.value),
          switch: parsePhraseTextarea(pauseSwitchEl.value),
        });
        applyPausePhrasePayload(data);
        const message = data.applied_live
          ? "Saved and applied to running session."
          : "Saved. Restart Robot Comic to apply.";
        setStatusMessage(pausePhrasesStatus, message, "ok");
      } catch (e) {
        setStatusMessage(pausePhrasesStatus, "Failed to save phrases.", "error");
      } finally {
        savePausePhrasesBtn.disabled = false;
      }
    });
  }

  if (resetPausePhrasesBtn) {
    resetPausePhrasesBtn.addEventListener("click", async () => {
      setStatusMessage(pausePhrasesStatus, "Resetting to defaults…");
      resetPausePhrasesBtn.disabled = true;
      try {
        pauseStopEl.value = "";
        pauseResumeEl.value = "";
        pauseShutdownEl.value = "";
        pauseSwitchEl.value = "";
        const data = await savePausePhrases({ stop: [], resume: [], shutdown: [], switch: [] });
        applyPausePhrasePayload(data);
        const message = data.applied_live
          ? "Reset to defaults and applied to running session."
          : "Reset to defaults. Restart Robot Comic to apply.";
        setStatusMessage(pausePhrasesStatus, message, "ok");
      } catch (e) {
        setStatusMessage(pausePhrasesStatus, "Failed to reset phrases.", "error");
      } finally {
        resetPausePhrasesBtn.disabled = false;
      }
    });
  }

  // Movement speed slider
  const speedSlider = document.getElementById("movement-speed-slider");
  const speedChip = document.getElementById("movement-speed-chip");
  const speedStatus = document.getElementById("movement-speed-status");

  function renderSpeedChip(value) {
    if (speedChip) speedChip.textContent = Number(value).toFixed(2) + "×";
  }

  (async () => {
    const data = await getMovementSpeed();
    if (!data || !data.ok) {
      if (speedSlider) speedSlider.disabled = true;
      setStatusMessage(speedStatus, "Movement manager unavailable.", "error");
      if (speedChip) speedChip.textContent = "off";
      return;
    }
    if (speedSlider) {
      speedSlider.min = String(data.min);
      speedSlider.max = String(data.max);
      speedSlider.step = String(data.step);
      speedSlider.value = String(data.value);
    }
    renderSpeedChip(data.value);
  })();

  if (speedSlider) {
    let debounceTimer = null;
    let lastSentValue = null;
    const commit = async () => {
      const value = parseFloat(speedSlider.value);
      if (Number.isNaN(value) || value === lastSentValue) return;
      lastSentValue = value;
      try {
        const data = await setMovementSpeed(value);
        if (data && data.ok) {
          renderSpeedChip(data.value);
          setStatusMessage(speedStatus, "");
        }
      } catch (e) {
        setStatusMessage(speedStatus, "Failed to update speed.", "error");
      }
    };
    speedSlider.addEventListener("input", () => {
      renderSpeedChip(speedSlider.value);
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(commit, 150);
    });
    speedSlider.addEventListener("change", () => {
      if (debounceTimer) clearTimeout(debounceTimer);
      commit();
    });
  }

  // Restart Comic admin button
  const restartBtn = document.getElementById("restart-app");
  const restartStatus = document.getElementById("restart-status");
  if (restartBtn) {
    restartBtn.addEventListener("click", async () => {
      if (!window.confirm("Restart Robot Comic now? The app will shut down gracefully and the autostart service should relaunch it.")) {
        return;
      }
      setStatusMessage(restartStatus, "Requesting restart…");
      restartBtn.disabled = true;
      try {
        const data = await restartApp();
        setStatusMessage(restartStatus, data.message || "Restart requested.", "ok");
      } catch (e) {
        restartBtn.disabled = false;
        setStatusMessage(restartStatus, "Restart hook unavailable.", "error");
      }
    });
  }

  // Handler for "Change API key" button
  changeKeyBtn.addEventListener("click", () => {
    editingCredentials = true;
    input.value = "";
    setStatusMessage(statusEl, "");
    renderCredentialPanels(st);
  });

  // Remove error styling when user starts typing
  input.addEventListener("input", () => {
    input.classList.remove("error");
  });
  hfHostCustom.addEventListener("input", () => {
    hfHostCustom.classList.remove("error");
    updateHFControls();
  });
  hfPort.addEventListener("input", () => {
    hfPort.classList.remove("error");
    updateHFControls();
  });
  hfMode.addEventListener("change", () => {
    hfHostCustom.classList.remove("error");
    hfPort.classList.remove("error");
    updateHFControls();
  });
  hfHostPreset.addEventListener("change", () => {
    hfHostCustom.classList.remove("error");
    updateHFControls();
  });
  localSttLanguage.addEventListener("input", () => localSttLanguage.classList.remove("error"));
  localSttCache.addEventListener("input", () => localSttCache.classList.remove("error"));
  localSttUpdate.addEventListener("input", () => localSttUpdate.classList.remove("error"));
  localSttResponse.addEventListener("change", () => {
    editingCredentials = false;
    setStatusMessage(statusEl, "");
    setSelectedLocalSTTOutput(localSttResponse.value);
    renderCredentialPanels(st);
  });
  localSttOutputInputs.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.disabled) return;
      setSelectedLocalSTTOutput(radio.value);
      setStatusMessage(statusEl, "");
      renderCredentialPanels(st);
    });
  });

  backendInputs.forEach((radio) => {
    radio.addEventListener("change", () => {
      editingCredentials = false;
      input.value = "";
      setSelectedBackend(radio.value);
      renderCredentialPanels(st);
    });
  });

  backendSaveBtn.addEventListener("click", async () => {
    setStatusMessage(backendStatusEl, `Saving ${backendMeta(selectedBackend).label}...`);
    try {
      const response = await saveBackendConfig(selectedBackend);
      setStatusMessage(backendStatusEl, response.message || "Saved. Reloading…", "ok");
      window.location.reload();
    } catch (e) {
      setStatusMessage(backendStatusEl, "Failed to save backend selection. Please try again.", "error");
    }
  });

  saveBtn.addEventListener("click", async () => {
    if (selectedBackend === HF_BACKEND || (selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === HF_BACKEND)) {
      const localMode = hfMode.value !== "deployed";
      setStatusMessage(statusEl, "Saving connection...");
      hfHostCustom.classList.remove("error");
      hfPort.classList.remove("error");

      try {
        if (localMode) {
          const host = resolveHFHost();
          const port = Number.parseInt((hfPort.value || "").trim(), 10);
          if (!host) {
            hfHostCustom.classList.add("error");
            setStatusMessage(statusEl, "Enter a valid host or IP address.", "warn");
            return;
          }
          if (!Number.isInteger(port) || port < 1 || port > 65535) {
            hfPort.classList.add("error");
            setStatusMessage(statusEl, "Enter a valid port between 1 and 65535.", "warn");
            return;
          }

          await saveBackendConfig(selectedBackend, {
            hfMode: "local",
            hfHost: host,
            hfPort: port,
          });
        } else {
          await saveBackendConfig(selectedBackend, {
            hfMode: "deployed",
          });
        }
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        if (e.message === "missing_hf_session_url") {
          setStatusMessage(
            statusEl,
            "The built-in Hugging Face server URL is unavailable. Restart the app and try again.",
            "error",
          );
        } else if (e.message === "empty_hf_host" || e.message === "invalid_hf_host") {
          hfHostCustom.classList.add("error");
          setStatusMessage(statusEl, "Enter a valid host or IP address.", "error");
        } else if (e.message === "invalid_hf_port") {
          hfPort.classList.add("error");
          setStatusMessage(statusEl, "Enter a valid port between 1 and 65535.", "error");
        } else {
          setStatusMessage(statusEl, "Failed to save the Hugging Face connection.", "error");
        }
      }
      if (selectedBackend === HF_BACKEND) return;
      setStatusMessage(statusEl, "Saved. Reloading…", "ok");
      window.location.reload();
      return;
    }

    if (selectedBackend === LOCAL_STT_BACKEND) {
      const language = (localSttLanguage.value || "").trim();
      const cacheDir = (localSttCache.value || "").trim();
      const updateInterval = Number.parseFloat((localSttUpdate.value || "").trim());
      localSttLanguage.classList.remove("error");
      localSttCache.classList.remove("error");
      localSttUpdate.classList.remove("error");
      if (!language || language.includes("/") || language.includes("\\")) {
        localSttLanguage.classList.add("error");
        setStatusMessage(statusEl, "Enter a valid language code, such as en.", "warn");
        return;
      }
      if (!cacheDir) {
        localSttCache.classList.add("error");
        setStatusMessage(statusEl, "Enter a writable model cache path.", "warn");
        return;
      }
      if (!Number.isFinite(updateInterval) || updateInterval < 0.1 || updateInterval > 2.0) {
        localSttUpdate.classList.add("error");
        setStatusMessage(statusEl, "Use an update interval from 0.1 to 2.0 seconds.", "warn");
        return;
      }
    }

    // Chatterbox needs no API key — save directly
    if (selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === CHATTERBOX_OUTPUT) {
      setStatusMessage(statusEl, "Saving Chatterbox config...");
      try {
        await saveBackendConfig(selectedBackend, {});
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        setStatusMessage(statusEl, "Failed to save Chatterbox config. Please try again.", "error");
      }
      return;
    }

    // ElevenLabs reads its API key + voice from dedicated fields, so don't fall
    // through to the main `#api-key` validation path.
    if (
      selectedBackend === LOCAL_STT_BACKEND
      && (localSttResponse.value === ELEVENLABS_OUTPUT || localSttResponse.value === LLAMA_ELEVENLABS_TTS_OUTPUT)
    ) {
      const elevenlabsKeyEl = document.getElementById("elevenlabs-key");
      const hasSavedKey = !!st.has_elevenlabs_key;
      const enteredKey = (elevenlabsKeyEl?.value || "").trim();
      if (!hasSavedKey && !enteredKey) {
        setStatusMessage(statusEl, "Enter your ELEVENLABS_API_KEY.", "warn");
        elevenlabsKeyEl?.classList.add("error");
        return;
      }
      elevenlabsKeyEl?.classList.remove("error");
      setStatusMessage(statusEl, "Saving ElevenLabs config...");
      try {
        await saveBackendConfig(selectedBackend, {});
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        if (e.message === "empty_key") {
          setStatusMessage(statusEl, "Enter your ELEVENLABS_API_KEY.", "warn");
          elevenlabsKeyEl?.classList.add("error");
        } else {
          setStatusMessage(statusEl, "Failed to save ElevenLabs config. Please try again.", "error");
        }
      }
      return;
    }

    const key = input.value.trim();
    if (!key) {
      setStatusMessage(statusEl, "Please enter a valid key.", "warn");
      input.classList.add("error");
      return;
    }
    const needsOpenAIValidation = (selectedBackend === OPENAI_BACKEND) ||
      (selectedBackend === LOCAL_STT_BACKEND && !localSttUsesGeminiTTS);
    setStatusMessage(statusEl, needsOpenAIValidation ? "Validating API key..." : "Saving token...");
    input.classList.remove("error");
    try {
      if (needsOpenAIValidation) {
        const validation = await validateKey(key);
        if (!validation.valid) {
          setStatusMessage(statusEl, "Invalid API key. Please check your key and try again.", "error");
          input.classList.add("error");
          return;
        }
        setStatusMessage(statusEl, "Key valid! Saving...", "ok");
      } else {
        setStatusMessage(statusEl, "Saving Gemini token...", "ok");
      }
      await saveBackendConfig(selectedBackend, { key });
      setStatusMessage(statusEl, "Saved. Reloading…", "ok");
      window.location.reload();
    } catch (e) {
      input.classList.add("error");
      if (needsOpenAIValidation && e.message === "invalid_api_key") {
        setStatusMessage(statusEl, "Invalid API key. Please check your key and try again.", "error");
      } else {
        setStatusMessage(
          statusEl,
          localSttUsesGeminiTTS || selectedBackend === GEMINI_BACKEND
            ? "Failed to save Gemini token. Please try again."
            : "Failed to validate/save key. Please try again.",
          "error",
        );
      }
    }
  });

  if (!(st.can_proceed ?? backendCanProceed(st, st.backend_provider || DEFAULT_BACKEND)) || st.requires_restart) {
    show(loading, false);
    return;
  }

  // Wait until backend routes are ready before rendering personalities UI
  const list = (await waitForPersonalityData()) || { choices: [] };
  setStatusMessage(statusEl, "");
  show(formPanel, false);
  if (!list.choices.length) {
    setStatusMessage(statusEl, "Personality endpoints not ready yet. Retry shortly.", "warn");
    show(loading, false);
    return;
  }

  // Initialize personalities UI
  try {
    const choices = Array.isArray(list.choices) ? list.choices : [];
    const DEFAULT_OPTION = choices[0] || "(built-in default)";
    const startupChoice = choices.includes(list.startup) ? list.startup : DEFAULT_OPTION;
    const currentChoice = choices.includes(list.current) ? list.current : startupChoice;

    function setStartupLabel(name) {
      const display = name && name !== DEFAULT_OPTION ? name : "Built-in default";
      pStartupLabel.textContent = `Launch on start: ${display}`;
    }

    // Populate select
    pSelect.innerHTML = "";
    for (const n of choices) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      pSelect.appendChild(opt);
    }
    if (choices.length) {
      const preferred = choices.includes(startupChoice) ? startupChoice : currentChoice;
      pSelect.value = preferred;
    }
    const voices = await getVoices();
    let currentVoice = await getCurrentVoice();
    pVoice.innerHTML = "";
    if (voices.length) {
      for (const v of voices) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        pVoice.appendChild(opt);
      }
    } else {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Backend default (recommended)";
      pVoice.appendChild(opt);
    }
    setStartupLabel(startupChoice);

    function renderToolCheckboxes(available, enabled) {
      pAvail.innerHTML = "";
      const enabledSet = new Set(enabled);
      for (const t of available) {
        const wrap = document.createElement("div");
        wrap.className = "chk";
        const id = `tool-${t}`;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.id = id;
        cb.value = t;
        cb.checked = enabledSet.has(t);
        const lab = document.createElement("label");
        lab.htmlFor = id;
        lab.textContent = t;
        wrap.appendChild(cb);
        wrap.appendChild(lab);
        pAvail.appendChild(wrap);
      }
    }

    function getSelectedTools() {
      const selected = new Set();
      pAvail.querySelectorAll('input[type="checkbox"]').forEach((el) => {
        if (el.checked) selected.add(el.value);
      });
      // Auto-include dependencies
      for (const [main, deps] of Object.entries(AUTO_WITH)) {
        if (selected.has(main)) {
          for (const d of deps) selected.add(d);
        }
      }
      return Array.from(selected);
    }

    function syncToolsTextarea() {
      const selected = getSelectedTools();
      const comments = pTools.value
        .split("\n")
        .filter((ln) => ln.trim().startsWith("#"));
      const body = selected.join("\n");
      pTools.value = (comments.join("\n") + (comments.length ? "\n" : "") + body).trim() + "\n";
    }

    pAvail.addEventListener("change", (ev) => {
      const target = ev.target;
      if (!(target instanceof HTMLInputElement) || target.type !== "checkbox") return;
      const name = target.value;
      if (AUTO_WITH[name]) {
        for (const dep of AUTO_WITH[name]) {
          const depEl = pAvail.querySelector(`input[value="${dep}"]`);
          if (depEl) depEl.checked = target.checked || depEl.checked;
        }
      }
      syncToolsTextarea();
    });

    async function loadSelected() {
      const selected = pSelect.value;
      const data = await loadPersonality(selected);
      pInstr.value = data.instructions || "";
      pTools.value = data.tools_text || "";
      const fallbackVoice = pVoice.options[0]?.value || "";
      const loadedVoice = voices.includes(data.voice) ? data.voice : fallbackVoice;
      const activeVoice = voices.includes(currentVoice) ? currentVoice : loadedVoice;
      pVoice.value = data.uses_default_voice ? activeVoice : loadedVoice;
      // Available tools as checkboxes
      renderToolCheckboxes(data.available_tools, data.enabled_tools);
      // Default name field to last segment of selection
      const idx = selected.lastIndexOf("/");
      pName.value = idx >= 0 ? selected.slice(idx + 1) : "";
      setStatusMessage(pStatus, `Loaded ${selected}`);
    }

    pSelect.addEventListener("change", loadSelected);
    await loadSelected();
    if (!voices.length) {
      setStatusMessage(pStatus, "Voices unavailable. The backend default voice will be used.", "warn");
    }
    show(personalityPanel, true);

    pApplyVoice.addEventListener("click", async () => {
      const voice = pVoice.value;
      if (!voice) return;
      setStatusMessage(pStatus, "Applying voice...");
      try {
        const res = await applyVoice(voice);
        currentVoice = voice;
        pVoice.value = voice;
        setStatusMessage(pStatus, res.status || `Voice changed to ${voice}.`, "ok");
      } catch (e) {
        setStatusMessage(pStatus, `Failed to apply voice${e.message ? ": " + e.message : ""}`, "error");
      }
    });

    pApply.addEventListener("click", async () => {
      setStatusMessage(pStatus, "Applying...");
      try {
        const res = await applyPersonality(pSelect.value);
        currentVoice = await getCurrentVoice();
        if (res.startup) setStartupLabel(res.startup);
        setStatusMessage(pStatus, res.status || "Applied.", "ok");
      } catch (e) {
        setStatusMessage(pStatus, `Failed to apply${e.message ? ": " + e.message : ""}`, "error");
      }
    });

    pPersist.addEventListener("click", async () => {
      setStatusMessage(pStatus, "Saving for startup...");
      try {
        const res = await applyPersonality(pSelect.value, { persist: true });
        currentVoice = await getCurrentVoice();
        if (res.startup) setStartupLabel(res.startup);
        setStatusMessage(pStatus, res.status || "Saved for startup.", "ok");
      } catch (e) {
        setStatusMessage(pStatus, `Failed to persist${e.message ? ": " + e.message : ""}`, "error");
      }
    });

    pNew.addEventListener("click", () => {
      pName.value = "";
      pInstr.value = "# Write your instructions here\n# e.g., Keep responses concise and friendly.";
      pTools.value = "# tools enabled for this profile\n";
      // Keep available tools list, clear selection
      pAvail.querySelectorAll('input[type="checkbox"]').forEach((el) => {
        el.checked = false;
      });
      pVoice.value = pVoice.options[0]?.value || "";
      setStatusMessage(pStatus, "Fill fields and click Save.");
    });

    pSave.addEventListener("click", async () => {
      const name = (pName.value || "").trim();
      if (!name) {
        setStatusMessage(pStatus, "Enter a valid name.", "warn");
        return;
      }
      setStatusMessage(pStatus, "Saving...");
      try {
        // Ensure tools.txt reflects checkbox selection and auto-includes
        syncToolsTextarea();
        const res = await savePersonality({
          name,
          instructions: pInstr.value || "",
          tools_text: pTools.value || "",
          voice: pVoice.value || pVoice.options[0]?.value || "",
        });
        // Refresh select choices
        pSelect.innerHTML = "";
        for (const n of res.choices) {
          const opt = document.createElement("option");
          opt.value = n;
          opt.textContent = n;
          if (n === res.value) opt.selected = true;
          pSelect.appendChild(opt);
        }
        setStatusMessage(pStatus, "Saved.", "ok");
        // Auto-apply
        try { await applyPersonality(pSelect.value); } catch {}
      } catch (e) {
        setStatusMessage(pStatus, "Failed to save.", "error");
      }
    });
  } catch (e) {
    setStatusMessage(statusEl, "UI failed to load. Please refresh.", "warn");
  } finally {
    // Hide loading when initial setup is done (regardless of key presence)
    show(loading, false);
  }
}

window.addEventListener("DOMContentLoaded", init);
