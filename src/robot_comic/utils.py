from __future__ import annotations
import os
import sys
import logging
import argparse
import warnings
import subprocess
from typing import TYPE_CHECKING, Optional

from reachy_mini import ReachyMini
from robot_comic.camera_worker import CameraWorker
from robot_comic.vision.head_tracking import HeadTracker


HEAD_TRACKER_ENV = "REACHY_MINI_HEAD_TRACKER"
HEAD_TRACKER_CHOICES = ("yolo", "mediapipe")
HEAD_TRACKER_DISABLED_VALUES = {"", "0", "false", "none", "off", "disabled"}


if TYPE_CHECKING:
    from robot_comic.vision.local_vision import VisionProcessor


class CameraVisionInitializationError(Exception):
    """Raised when camera or vision setup fails in an expected way."""


def get_requested_head_tracker(args: argparse.Namespace) -> str | None:
    """Return the requested head-tracking backend from CLI args or environment.

    Priority: CLI flag > REACHY_MINI_HEAD_TRACKER env var > default (mediapipe).
    Set REACHY_MINI_HEAD_TRACKER=off to disable head tracking without a CLI flag.
    """
    cli_value = getattr(args, "head_tracker", None)
    if cli_value is not None:
        return str(cli_value)

    raw_env = os.getenv(HEAD_TRACKER_ENV)
    if raw_env is None:
        return "mediapipe"  # default: mediapipe on when no env override

    env_value = raw_env.strip().lower()
    if env_value in HEAD_TRACKER_DISABLED_VALUES:
        return None
    if env_value in HEAD_TRACKER_CHOICES:
        return env_value
    raise CameraVisionInitializationError(
        f"Invalid {HEAD_TRACKER_ENV}={raw_env!r}. Expected one of: {', '.join(HEAD_TRACKER_CHOICES)}, or a disabled value (off, false, 0).",
    )


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Robot Comic")
    parser.add_argument(
        "--head-tracker",
        choices=HEAD_TRACKER_CHOICES,
        default=None,
        help=(
            "Optional head-tracking backend: yolo uses a local face detector in a subprocess, "
            f"mediapipe uses reachy_mini_toolbox in process. Disabled by default. "
            f"Can also be set with {HEAD_TRACKER_ENV}=mediapipe."
        ),
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of the selected realtime backend vision",
    )
    parser.add_argument(
        "--sim",
        "--gradio",
        dest="sim",
        default=False,
        action="store_true",
        help="Run in simulation/dev mode: serve the browser audio chat (FastRTC) at /chat and the admin UI at /. Required when running against the Reachy Mini simulator. --gradio is accepted as a deprecated alias.",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    args, extras = parser.parse_known_args()
    # The legacy --gradio alias keeps existing launchers and scripts working,
    # but warn so users (and the next code-search) migrate to --sim.
    if "--gradio" in sys.argv[1:]:
        warnings.warn(
            "--gradio is deprecated; use --sim instead. The alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        logging.getLogger(__name__).warning(
            "--gradio is deprecated; use --sim instead (will be removed in a future release)."
        )
    return args, extras


def initialize_camera_and_vision(
    args: argparse.Namespace,
    current_robot: ReachyMini,
) -> tuple[CameraWorker | None, VisionProcessor | None]:
    """Initialize camera capture, optional head tracking, and optional local vision."""
    camera_worker: Optional[CameraWorker] = None
    head_tracker: HeadTracker | None = None
    vision_processor: Optional[VisionProcessor] = None

    if not args.no_camera:
        requested_head_tracker = get_requested_head_tracker(args)
        if requested_head_tracker is not None:
            try:
                if requested_head_tracker == "yolo":
                    from robot_comic.vision.head_tracking.yolo_process import (
                        YoloHeadTrackerProcess,
                    )

                    head_tracker = YoloHeadTrackerProcess()
                    logging.getLogger(__name__).info("Using yolo head tracker subprocess")
                else:
                    from robot_comic.vision.head_tracking.mediapipe import (
                        MediapipeHeadTracker,
                    )

                    head_tracker = MediapipeHeadTracker()
                    logging.getLogger(__name__).info("Using mediapipe head tracker in process")
            except Exception as e:
                raise CameraVisionInitializationError(
                    f"Failed to initialize {requested_head_tracker} head tracker: {e}",
                ) from e

        camera_worker = CameraWorker(current_robot, head_tracker)

        if args.local_vision:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from robot_comic.vision.local_vision import VisionProcessor",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode < 0:
                raise CameraVisionInitializationError(
                    "Local vision import crashed on this machine. "
                    "Run without --local-vision or install compatible dependencies.",
                )
            try:
                from robot_comic.vision.local_vision import initialize_vision_processor

            except ImportError as e:
                raise CameraVisionInitializationError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e

            vision_processor = initialize_vision_processor()
        else:
            logging.getLogger(__name__).info(
                "Using the selected realtime backend for vision (default). Use --local-vision for local processing.",
            )

    return camera_worker, vision_processor


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
