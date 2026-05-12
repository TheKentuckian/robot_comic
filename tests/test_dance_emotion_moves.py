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
