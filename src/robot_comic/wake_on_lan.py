"""Wake-on-LAN magic packet sender.

Sends a standard WoL magic packet over UDP so that a sleeping host
(e.g. the laptop running llama-server or Chatterbox) can be woken
before the app retries a connection.

The magic packet is: 6 × 0xFF followed by 16 repetitions of the
target MAC address (102 bytes total), sent to the broadcast address
on UDP port 9.

Only stdlib is required — no external dependencies.
"""

import socket


def _parse_mac(mac_address: str) -> bytes:
    """Parse a MAC address string into a 6-byte sequence.

    Accepted formats (case-insensitive)::

        aa:bb:cc:dd:ee:ff
        aa-bb-cc-dd-ee-ff
        aabbccddeeff

    Returns:
        6-byte :class:`bytes` object.

    Raises:
        ValueError: If the MAC address is malformed.

    """
    # Normalise separators and strip surrounding whitespace.
    clean = mac_address.strip().replace(":", "").replace("-", "")
    if len(clean) != 12:
        raise ValueError(
            f"Invalid MAC address {mac_address!r}: expected 12 hex digits "
            f"(with optional ':' or '-' separators), got {len(clean)} hex chars."
        )
    try:
        mac_bytes = bytes.fromhex(clean)
    except ValueError as exc:
        raise ValueError(f"Invalid MAC address {mac_address!r}: {exc}") from exc
    return mac_bytes


def send_magic_packet(
    mac_address: str,
    broadcast: str = "255.255.255.255",
    port: int = 9,
) -> None:
    """Send a Wake-on-LAN magic packet to *mac_address*.

    The packet is the standard WoL format:
    - 6 bytes of ``0xFF``
    - 16 repetitions of the 6-byte MAC address
    = 102 bytes total, sent as a single UDP datagram.

    Args:
        mac_address: Target MAC address in ``aa:bb:cc:dd:ee:ff``,
            ``aa-bb-cc-dd-ee-ff``, or ``aabbccddeeff`` format
            (case-insensitive).
        broadcast: Broadcast/directed-broadcast address to send to.
            Defaults to ``255.255.255.255``.
        port: UDP port.  The WoL standard uses port 9 (discard).

    Raises:
        ValueError: If *mac_address* is malformed.
        OSError: If the UDP socket cannot be created or the packet
            cannot be sent.

    """
    mac_bytes = _parse_mac(mac_address)
    magic_packet = b"\xff" * 6 + mac_bytes * 16

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(magic_packet, (broadcast, port))
