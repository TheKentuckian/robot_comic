from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_recorded_moves(duration: float = 2.0):
    fake_move = MagicMock()
    fake_move.duration = duration
    eye4 = np.eye(4, dtype=np.float64)
    fake_move.evaluate.side_effect = lambda t: (eye4, (0.0, 0.0), 0.0)
    recorded = MagicMock()
    recorded.get.return_value = fake_move
    return recorded, fake_move


def test_emotion_queue_move_duration_scaled():
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    recorded, _ = _make_recorded_moves(duration=2.0)
    move = EmotionQueueMove("laughing1", recorded, speed_factor=0.5)
    assert move.duration == pytest.approx(4.0)  # 2.0 / 0.5


def test_emotion_queue_move_evaluate_scales_t():
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    recorded, fake_move = _make_recorded_moves(duration=2.0)
    move = EmotionQueueMove("laughing1", recorded, speed_factor=2.0)
    move.evaluate(1.0)
    fake_move.evaluate.assert_called_once_with(2.0)  # t * speed_factor = 1.0 * 2.0


def test_emotion_queue_move_default_speed_factor_1():
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    recorded, fake_move = _make_recorded_moves(duration=2.0)
    move = EmotionQueueMove("laughing1", recorded)
    assert move.duration == pytest.approx(2.0)
    move.evaluate(0.5)
    fake_move.evaluate.assert_called_once_with(0.5)


def test_dance_queue_move_duration_scaled():
    from robot_comic.dance_emotion_moves import DanceQueueMove

    fake_dance = MagicMock()
    fake_dance.duration = 3.0
    eye4 = np.eye(4, dtype=np.float64)
    fake_dance.evaluate.side_effect = lambda t: (eye4, (0.0, 0.0), 0.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "robot_comic.dance_emotion_moves.DanceMove",
            lambda name: fake_dance,
        )
        move = DanceQueueMove("robot_groove", speed_factor=0.6)
    assert move.duration == pytest.approx(3.0 / 0.6, rel=1e-3)


def test_goto_queue_move_duration_scaled():
    """speed_factor < 1 should stretch duration so motion completes slower."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    target = np.eye(4, dtype=np.float32)
    move = GotoQueueMove(target_head_pose=target, duration=1.0, speed_factor=0.5)
    assert move.duration == pytest.approx(2.0)


def test_goto_queue_move_default_speed_factor_1():
    """Default speed_factor leaves the requested duration untouched."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    target = np.eye(4, dtype=np.float32)
    move = GotoQueueMove(target_head_pose=target, duration=1.5)
    assert move.duration == pytest.approx(1.5)


def test_goto_queue_move_speed_factor_clamped():
    """Speed factors outside [0.1, 2.0] are clamped to the bracket."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    target = np.eye(4, dtype=np.float32)
    move = GotoQueueMove(target_head_pose=target, duration=1.0, speed_factor=5.0)
    assert move.speed_factor == pytest.approx(2.0)
    assert move.duration == pytest.approx(0.5)


def test_goto_queue_move_linear_by_default():
    """Without ease=True the goto move uses straight linear interpolation."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    target = np.eye(4, dtype=np.float32)
    move = GotoQueueMove(
        target_head_pose=target,
        target_antennas=(10.0, 20.0),
        start_antennas=(0.0, 0.0),
        duration=1.0,
    )
    # At t=0.5 with linear interpolation, antennas should land at the midpoint.
    _, antennas, _ = move.evaluate(0.5)
    assert antennas is not None
    assert antennas[0] == pytest.approx(5.0)
    assert antennas[1] == pytest.approx(10.0)


def test_goto_queue_move_smoothstep_when_ease_enabled():
    """ease=True applies cubic smoothstep (#264) so velocity is zero at endpoints."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    target = np.eye(4, dtype=np.float32)
    move = GotoQueueMove(
        target_head_pose=target,
        target_antennas=(10.0, 20.0),
        start_antennas=(0.0, 0.0),
        duration=1.0,
        ease=True,
    )
    # Smoothstep midpoint: 3*0.5² − 2*0.5³ = 0.75 − 0.25 = 0.5 → same numeric
    # midpoint as linear, but the shape diverges off-centre. Check t=0.25:
    # smoothstep(0.25) = 3*0.0625 − 2*0.015625 = 0.1875 − 0.03125 = 0.15625.
    _, antennas, _ = move.evaluate(0.25)
    assert antennas is not None
    assert antennas[0] == pytest.approx(10.0 * 0.15625)
    assert antennas[1] == pytest.approx(20.0 * 0.15625)


def test_goto_queue_move_smoothstep_endpoints_match_linear():
    """Smoothstep maps 0→0 and 1→1, so endpoints are unchanged."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    target = np.eye(4, dtype=np.float32)
    move = GotoQueueMove(
        target_head_pose=target,
        target_antennas=(10.0, 20.0),
        start_antennas=(0.0, 0.0),
        duration=1.0,
        ease=True,
    )
    _, antennas_start, _ = move.evaluate(0.0)
    _, antennas_end, _ = move.evaluate(1.0)
    assert antennas_start is not None and antennas_end is not None
    assert antennas_start[0] == pytest.approx(0.0)
    assert antennas_end[0] == pytest.approx(10.0)


def test_movement_manager_set_speed_factor_clamps():
    from robot_comic.moves import MovementManager

    mm = MovementManager.__new__(MovementManager)
    mm.speed_factor = 1.0
    mm.set_speed_factor(5.0)
    assert mm.speed_factor == pytest.approx(2.0)
    mm.set_speed_factor(0.0)
    assert mm.speed_factor == pytest.approx(0.1)
    mm.set_speed_factor(0.6)
    assert mm.speed_factor == pytest.approx(0.6)
