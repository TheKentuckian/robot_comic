"""Entrypoint for Robot Comic."""

import warnings


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)

import os
import sys
import time
import signal
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream

from reachy_mini import ReachyMini, ReachyMiniApp
from robot_comic.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    get_requested_head_tracker,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for Robot Comic."""
    app = RobotComic()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run Robot Comic."""
    # Putting these dependencies here makes the dashboard faster to load when Robot Comic is installed.
    from robot_comic.moves import MovementManager
    from robot_comic.config import (
        HF_BACKEND,
        OPENAI_BACKEND,
        CHATTERBOX_OUTPUT,
        GEMINI_TTS_OUTPUT,
        LOCAL_STT_BACKEND,
        HF_LOCAL_CONNECTION_MODE,
        config,
        is_gemini_model,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
    from robot_comic.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)
    try:
        import subprocess
        _git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        _git_hash = "unknown"
    logger.info("Starting Robot Comic (git: %s)", _git_hash)
    startup_settings = StartupSettings()

    if instance_path is not None:
        try:
            from dotenv import load_dotenv

            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
        except Exception as e:
            logger.warning("Failed to load instance configuration: %s", e)

        try:
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except Exception as e:
            logger.warning("Failed to load startup settings: %s", e)

    if config.BACKEND_PROVIDER == HF_BACKEND:
        logger.info(
            "Configured backend provider: %s (%s), connection mode: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            get_hf_connection_selection().mode,
        )
    else:
        logger.info(
            "Configured backend provider: %s (%s), model: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            config.MODEL_NAME,
        )

    from robot_comic.pause import PauseController
    from robot_comic.console import LocalStream
    from robot_comic.pause_settings import read_pause_settings
    from robot_comic.tools.core_tools import ToolDependencies
    from robot_comic.audio.head_wobbler import HeadWobbler

    if args.no_camera and get_requested_head_tracker(args) is not None:
        logger.warning("Head tracking disabled: --no-camera flag is set. Remove --no-camera to enable head tracking.")

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.sim:
        logger.info("Simulation mode detected. Automatically enabling --sim.")
        args.sim = True

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(args, robot)
    except CameraVisionInitializationError as e:
        logger.error("Failed to initialize camera/vision: %s", e)
        sys.exit(1)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    def _request_shutdown_from_pause() -> None:
        logger.info("Pause controller requested shutdown")
        if app_stop_event is not None:
            app_stop_event.set()
        else:
            logger.warning("No app_stop_event available; cannot trigger graceful shutdown")

    pause_settings = read_pause_settings(instance_path)
    pause_controller = PauseController(
        clear_move_queue=movement_manager.clear_move_queue,
        on_shutdown=_request_shutdown_from_pause,
        stop_phrases=pause_settings.resolved_stop(),
        resume_phrases=pause_settings.resolved_resume(),
        shutdown_phrases=pause_settings.resolved_shutdown(),
        switch_phrases=pause_settings.resolved_switch(),
    )

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_processor=vision_processor,
        head_wobbler=head_wobbler,
        pause_controller=pause_controller,
        instance_path=Path(instance_path) if instance_path is not None else None,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    if is_gemini_model():
        from robot_comic.gemini_live import GeminiLiveHandler

        logger.info(
            "Using %s via GeminiLiveHandler",
            get_backend_label(config.BACKEND_PROVIDER),
        )
        handler = GeminiLiveHandler(
            deps,
            sim_mode=args.sim,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )
    elif config.BACKEND_PROVIDER == HF_BACKEND:
        from robot_comic.huggingface_realtime import HuggingFaceRealtimeHandler

        hf_connection_selection = get_hf_connection_selection()
        transport_label = (
            "Hugging Face direct websocket"
            if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
            else "Hugging Face session proxy"
        )
        logger.info(
            "Using %s via Hugging Face realtime handler (%s)",
            get_backend_label(config.BACKEND_PROVIDER),
            transport_label,
        )
        handler = HuggingFaceRealtimeHandler(
            deps,
            sim_mode=args.sim,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )  # type: ignore[assignment]
    elif config.BACKEND_PROVIDER == LOCAL_STT_BACKEND:
        from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler
        from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler
        from robot_comic.local_stt_realtime import (
            LocalSTTOpenAIRealtimeHandler,
            LocalSTTHuggingFaceRealtimeHandler,
        )

        local_stt_response_backend = getattr(config, "LOCAL_STT_RESPONSE_BACKEND", OPENAI_BACKEND)
        logger.info("Using %s", get_backend_label(config.BACKEND_PROVIDER))

        if local_stt_response_backend == HF_BACKEND:
            handler_class = LocalSTTHuggingFaceRealtimeHandler
        elif local_stt_response_backend == GEMINI_TTS_OUTPUT:
            handler_class = LocalSTTGeminiTTSHandler
        elif local_stt_response_backend == CHATTERBOX_OUTPUT:
            handler_class = LocalSTTChatterboxHandler
        else:
            handler_class = LocalSTTOpenAIRealtimeHandler

        handler = handler_class(
            deps,
            sim_mode=args.sim,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )  # type: ignore[assignment]
    else:
        from robot_comic.openai_realtime import OpenaiRealtimeHandler

        logger.info(
            "Using %s via OpenAI realtime handler (OpenAI Realtime API)",
            get_backend_label(config.BACKEND_PROVIDER),
        )
        handler = OpenaiRealtimeHandler(
            deps,
            sim_mode=args.sim,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )  # type: ignore[assignment]

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.sim:
        # Sim/dev mode: bridge audio to the browser via FastRTC and serve the
        # same static admin UI at "/" that the on-robot headless build uses.
        # The FastRTC chat UI lives at "/chat" so it doesn't collide with the
        # admin UI.
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        # Mount the static admin UI (and its REST endpoints) by piggy-backing
        # on a settings-only LocalStream. It never starts an audio pipeline
        # because we don't call .launch() on it.
        admin_only_stream = LocalStream(
            handler=None,
            robot=None,
            settings_app=app,
            instance_path=instance_path,
            app_stop_event=app_stop_event,
            pause_controller=pause_controller,
            movement_manager=movement_manager,
        )
        admin_only_stream.init_admin_ui()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[chatbot],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Robot Comic"},
        )
        stream_manager = stream.ui

        app = gr.mount_gradio_app(app, stream.ui, path="/chat")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
            app_stop_event=app_stop_event,
            pause_controller=pause_controller,
            movement_manager=movement_manager,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    # Translate SIGTERM into the same graceful path as KeyboardInterrupt so
    # `systemctl stop` (and any future power-button hook) drives the head back
    # to a neutral pose instead of letting motors cut mid-motion.
    def _request_graceful_shutdown(signum: int, _frame: Any) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        logger.info("Received %s; requesting graceful shutdown", name)
        if app_stop_event is not None:
            app_stop_event.set()
        else:
            # No external stop_event plumbed — fall back to raising
            # KeyboardInterrupt so the existing finally-block path runs.
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _request_graceful_shutdown)
    except ValueError:
        # signal.signal() only works in the main thread; skip silently if not.
        logger.debug("Skipped SIGTERM handler install (not in main thread)")

    try:
        if args.sim:
            stream_manager.launch(server_name="0.0.0.0")
        else:
            stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class RobotComic(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for Robot Comic."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run Robot Comic."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = RobotComic()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
