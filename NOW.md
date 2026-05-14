# Now

## Current focus

Infrastructure polish and persona completeness. The core comedian pipeline is
stable; current effort is on reliability (boot sequencing, hardware safety,
echo suppression) and rounding out the remaining persona-specific assets and
guardrails.

## In progress

- [#155] wake_up animation physical smoothing — IK solver / motion profile gentling
- [#138] test: admin Restart end-to-end click-test (exit-75 → systemd relaunch)
- [#121] boot: investigate 11.5s gap between movement manager init and handler ready
- [#108] Welcome Gate: finish remaining welcome_name.wav assets (5 comedians + 6 characters)

## Next up

- [#54] Modular audio pipeline — separate `AUDIO_INPUT_BACKEND` + `AUDIO_OUTPUT_BACKEND` config
- [#78] Benchmark Qwen3 14B dense vs 35B MoE for production suitability
- [#20] Basic L-R movement testing
- [#19] Servo calibration: avoid body collisions

## Recently shipped

*(rolling 30-day snapshot — last updated 2026-05-13)*

- [#110] (PR #204) Welcome Gate state machine gates handler until persona name is heard
- [#31]  (PR #203) Boot: send Wake-on-LAN magic packet when llama-server is asleep
- [#91]  (PR #202) Echo suppression: tighter playback-end estimate
- [#77]  (PR #201) Startup: pre-warm llama-server KV cache to cut first-turn latency
- [#113] (PR #200) Warmup: Windows winsound player + optional fast-blip cue
- [#53]  (PR #199) Chatterbox: auto-normalize voice clone output gain
- [#42]  (PR #198) Bill Hicks disengagement guardrail
- [#41]  (PR #197) Startup: optional kiosk-mode voice prompt before persona selection
- [#132] (PR #196) Battery: surface Reachy battery status in admin UI and monitor
- [#63]  (PR #195) Profiles: complete George Carlin persona
- [#21]  (PR #194) STT: Moonshine .ort fast-load + page-cache prewarm
- [#34]  (PR #193) Profiles: complete Rodney Dangerfield persona
- [#8]   (PR #191) Persona: persist joke history across sessions to avoid repeats
- [#135] (PR #189) Gemini Live: exponential-backoff presence check on user silence
- [#139] (PR #188) Profiles: add gemini_live.txt delivery styling for each persona
- [#6]   (PR #187) Boot: replace 30s sleep with active Reachy-daemon ready-poll
- [#98]  (PR #186) Trigger: switch primary stop-word prefix to 'robot' with backward compat
- [#93]  (PR #185) Monitor: show transcript excerpt on pending row

---

*Updated: 2026-05-13*
