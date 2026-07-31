"""IP and TCP checksum calculation utilities.

These functions compute ones-complement checksums as defined in RFC 791
(IP) and RFC 793 (TCP) and are used to repair checksums in adversarially
perturbed packets so that they remain structurally valid.

Notes
-----
Both functions work on raw byte sequences represented as lists of integers
(0–255), matching the representation used in the flow image tensors.
"""


def _transformed_hex_preprocess(decimal: int) -> str:
    """Convert *decimal* to a zero-padded, even-length hex string."""
    hexa = hex(decimal)[2:]  # strip '0x'
    if len(hexa) % 2 != 0:
        hexa = "0" + hexa
    return hexa


def calc_checksum_ip(integer_values: list[int]) -> list[int]:
    """Compute the RFC 791 IP header checksum.

    Parameters
    ----------
    integer_values : list of int
        The IP header bytes (20 bytes) as integers.  The checksum field
        (bytes 10–11) should be zeroed out before calling this function.

    Returns
    -------
    list of int
        Two-byte checksum as a list ``[high_byte, low_byte]``.
    """
    byte_values = bytes(integer_values)
    checksum_header = byte_values.hex()

    # Ones-complement sum over 16-bit words
    total = 0
    for i in range(0, len(checksum_header), 4):
        word = checksum_header[i : i + 4]
        total += int(word, 16)

    # Fold carries
    ch = hex(total)[2:]
    if len(ch) % 4 != 0:
        ch = "000" + ch
    folded = int(ch[:4], 16) + int(ch[4:], 16)
    check_hex = hex(folded)[2:]

    if len(check_hex) > 4:
        check_hex = check_hex.zfill(8)
        check_hex = _transformed_hex_preprocess(
            int(check_hex[:4], 16) + int(check_hex[4:], 16)
        )

    checksum_dec = int("ffff", 16) - int(check_hex, 16)

    if checksum_dec != 0:
        check_sum = hex(checksum_dec)[2:]
        if len(check_sum) % 4 != 0:
            check_sum = check_sum.zfill(4)
    else:
        check_sum = "0000"

    byte_result = bytes.fromhex(check_sum)
    return list(byte_result)


def calc_tcp_checksum(
    ip_header: str,
    tcp_header: str,
    payload_header: str,
) -> str:
    """Compute the RFC 793 TCP checksum.

    Parameters
    ----------
    ip_header : str
        Hex string of the 20-byte IP header (used to build the pseudo-header).
    tcp_header : str
        Hex string of the TCP header with checksum field zeroed out.
    payload_header : str
        Hex string of the TCP payload (may be empty string).

    Returns
    -------
    str
        4-character hex string representing the 2-byte TCP checksum.
    """
    # Zero out TCP checksum field (bytes 16–17 of TCP header = chars 32–35)
    tcp_no_check = tcp_header[:32] + tcp_header[36:]

    # Sum TCP header words
    sum_header = 0
    for i in range(0, len(tcp_no_check), 4):
        word = tcp_no_check[i : i + 4]
        sum_header += int(word, 16)

    # Pad payload to 4-char alignment
    payload = payload_header
    if len(payload) % 4 != 0:
        pad_len = 4 - len(payload) % 4
        int_fg = len(payload) // 4
        payload = payload[: int_fg * 4] + "0" * pad_len + payload[int_fg * 4 :]

    sum_payload = 0
    for i in range(0, len(payload), 4):
        word = payload[i : i + 4]
        sum_payload += int(word, 16)

    # TCP segment length from IP total-length field
    total_ip_length = int(ip_header[4:8], 16) - 20  # subtract IP header size

    # Pseudo-header contribution: src IP, dst IP, zero + proto (6), TCP length
    src_ip = int(ip_header[24:28], 16) + int(ip_header[28:32], 16)
    dst_ip = int(ip_header[32:36], 16) + int(ip_header[36:], 16)
    tcp_check_dec = (
        src_ip + dst_ip + total_ip_length + sum_header + sum_payload + 6
    )

    # Fold carries
    ch = hex(tcp_check_dec)[2:]
    if len(ch) % 4 != 0:
        ch = ch.zfill(4 * ((len(ch) // 4) + 1))
    folded = int(ch[4:], 16) + int(ch[:4], 16)
    check_hex = hex(folded)[2:]

    if len(check_hex) > 4:
        check_hex = check_hex.zfill(8)
        check_hex = _transformed_hex_preprocess(
            int(check_hex[:4], 16) + int(check_hex[4:], 16)
        )

    checksum_dec = int("ffff", 16) - int(check_hex, 16)

    if checksum_dec > 0:
        check_sum = hex(checksum_dec)[2:]
        if len(check_sum) % 4 != 0:
            check_sum = check_sum.zfill(4)
    else:
        check_sum = "0000"

    return check_sum
