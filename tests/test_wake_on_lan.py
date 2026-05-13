"""Unit tests for the Wake-on-LAN magic packet sender."""

import socket
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.wake_on_lan import _parse_mac, send_magic_packet


# ---------------------------------------------------------------------------
# MAC parsing
# ---------------------------------------------------------------------------


def test_parse_mac_colon_separated() -> None:
    """Colon-separated MAC is parsed to 6 bytes."""
    result = _parse_mac("aa:bb:cc:dd:ee:ff")
    assert result == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])


def test_parse_mac_hyphen_separated() -> None:
    """Hyphen-separated MAC is parsed to 6 bytes."""
    result = _parse_mac("aa-bb-cc-dd-ee-ff")
    assert result == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])


def test_parse_mac_no_separator() -> None:
    """Unseparated hex MAC is parsed to 6 bytes."""
    result = _parse_mac("AABBCCDDEEFF")
    assert result == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])


def test_parse_mac_case_insensitive() -> None:
    """MAC parsing is case-insensitive."""
    lower = _parse_mac("aa:bb:cc:dd:ee:ff")
    upper = _parse_mac("AA:BB:CC:DD:EE:FF")
    mixed = _parse_mac("Aa:Bb:Cc:Dd:Ee:Ff")
    assert lower == upper == mixed


def test_parse_mac_rejects_too_short() -> None:
    """A MAC with fewer than 12 hex chars raises ValueError."""
    with pytest.raises(ValueError, match="Invalid MAC"):
        _parse_mac("aa:bb:cc:dd:ee")


def test_parse_mac_rejects_too_long() -> None:
    """A MAC with more than 12 hex chars raises ValueError."""
    with pytest.raises(ValueError, match="Invalid MAC"):
        _parse_mac("aa:bb:cc:dd:ee:ff:00")


def test_parse_mac_rejects_non_hex() -> None:
    """A MAC with non-hex characters raises ValueError."""
    with pytest.raises(ValueError, match="Invalid MAC"):
        _parse_mac("zz:bb:cc:dd:ee:ff")


# ---------------------------------------------------------------------------
# Magic packet payload
# ---------------------------------------------------------------------------

_EXPECTED_HEADER = b"\xff" * 6
_TEST_MAC = "aa:bb:cc:dd:ee:ff"
_TEST_MAC_BYTES = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
_EXPECTED_PAYLOAD = _EXPECTED_HEADER + _TEST_MAC_BYTES * 16


def _capture_sendto(mac: str, broadcast: str = "255.255.255.255", port: int = 9) -> bytes:
    """Call send_magic_packet with a mocked socket and return the sent bytes."""
    mock_sock = MagicMock()
    mock_sock.__enter__ = MagicMock(return_value=mock_sock)
    mock_sock.__exit__ = MagicMock(return_value=False)

    with patch("robot_comic.wake_on_lan.socket.socket", return_value=mock_sock):
        send_magic_packet(mac, broadcast=broadcast, port=port)

    mock_sock.sendto.assert_called_once()
    sent_bytes, addr = mock_sock.sendto.call_args[0]
    return sent_bytes, addr  # type: ignore[return-value]


def test_magic_packet_length() -> None:
    """Magic packet is exactly 102 bytes (6 header + 16×6 MAC)."""
    sent, _ = _capture_sendto(_TEST_MAC)
    assert len(sent) == 102


def test_magic_packet_header() -> None:
    """First 6 bytes of the packet are all 0xFF."""
    sent, _ = _capture_sendto(_TEST_MAC)
    assert sent[:6] == _EXPECTED_HEADER


def test_magic_packet_mac_repeated_16_times() -> None:
    """Bytes 6–102 are the MAC address repeated 16 times."""
    sent, _ = _capture_sendto(_TEST_MAC)
    assert sent[6:] == _TEST_MAC_BYTES * 16


def test_magic_packet_full_payload() -> None:
    """Entire packet matches the canonical WoL magic packet format."""
    sent, _ = _capture_sendto(_TEST_MAC)
    assert sent == _EXPECTED_PAYLOAD


def test_magic_packet_accepts_hyphen_format() -> None:
    """Hyphen-separated MAC produces an identical packet to colon-separated."""
    sent_colon, _ = _capture_sendto("aa:bb:cc:dd:ee:ff")
    sent_hyphen, _ = _capture_sendto("aa-bb-cc-dd-ee-ff")
    assert sent_colon == sent_hyphen


def test_magic_packet_accepts_no_separator_format() -> None:
    """Unseparated MAC produces an identical packet to colon-separated."""
    sent_colon, _ = _capture_sendto("aa:bb:cc:dd:ee:ff")
    sent_plain, _ = _capture_sendto("AABBCCDDEEFF")
    assert sent_colon == sent_plain


# ---------------------------------------------------------------------------
# sendto call shape
# ---------------------------------------------------------------------------


def test_sendto_destination_address() -> None:
    """Packet is sent to the specified broadcast address and port."""
    _, addr = _capture_sendto(_TEST_MAC, broadcast="192.168.1.255", port=9)
    assert addr == ("192.168.1.255", 9)


def test_sendto_default_broadcast() -> None:
    """Default broadcast address is 255.255.255.255 on port 9."""
    _, addr = _capture_sendto(_TEST_MAC)
    assert addr == ("255.255.255.255", 9)


def test_socket_broadcast_option_set() -> None:
    """SO_BROADCAST socket option is enabled before sending."""
    mock_sock = MagicMock()
    mock_sock.__enter__ = MagicMock(return_value=mock_sock)
    mock_sock.__exit__ = MagicMock(return_value=False)

    with patch("robot_comic.wake_on_lan.socket.socket", return_value=mock_sock):
        send_magic_packet(_TEST_MAC)

    mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_send_magic_packet_raises_on_bad_mac() -> None:
    """send_magic_packet propagates ValueError for malformed MAC addresses."""
    with pytest.raises(ValueError, match="Invalid MAC"):
        send_magic_packet("not-a-mac")
