"""Read-only Garmin FIT timer / average-power reconciliation without dependencies."""
from __future__ import annotations

import io
import zipfile
from typing import Any

_CRC = (0, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401, 0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400)
_SIGNED = {1, 3, 5, 14}
_INTEGER = {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16}


def _crc(data: bytes) -> int:
    value = 0
    for byte in data:
        value = ((value >> 4) ^ _CRC[value & 15] ^ _CRC[byte & 15]) & 0xFFFF
        value = ((value >> 4) ^ _CRC[value & 15] ^ _CRC[(byte >> 4) & 15]) & 0xFFFF
    return value


def _extract_fit(original: bytes) -> bytes:
    if original[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".fit") and not m.is_dir()]
            if len(members) != 1:
                raise ValueError("Expected one FIT file in activity archive")
            if members[0].file_size > 25_000_000:
                raise ValueError("FIT exceeds supported safety limit")
            return archive.read(members[0])
    return original


def _integer(raw: bytes, base: int, endian: str) -> int | None:
    kind = base & 31
    if kind not in _INTEGER or len(raw) not in (1, 2, 4, 8):
        return None
    signed = kind in _SIGNED
    value = int.from_bytes(raw, endian, signed=signed)
    invalid = (1 << (len(raw) * 8 - 1)) - 1 if signed else (1 << (len(raw) * 8)) - 1
    return None if value == invalid else value


def audit_fit_bytes(original: bytes, activity_id: str | None = None) -> dict[str, Any]:
    """Use FIT timer events and both zero-inclusion policies to verify Garmin averages.

    This does not claim to validate independent left/right power sensing or all
    physiology. It deliberately never replaces Garmin's official session metrics.
    """
    fit = _extract_fit(original)
    if len(fit) < 14 or fit[8:12] != b".FIT":
        raise ValueError("Invalid FIT header")
    start = fit[0]
    if start not in (12, 14):
        raise ValueError("Unsupported FIT header size")
    end = start + int.from_bytes(fit[4:8], "little")
    if end + 2 > len(fit):
        raise ValueError("Truncated FIT")
    if start == 14 and _crc(fit[:12]) != int.from_bytes(fit[12:14], "little"):
        raise ValueError("FIT header CRC invalid")
    if _crc(fit[:end]) != int.from_bytes(fit[end:end + 2], "little"):
        raise ValueError("FIT data CRC invalid")
    definitions: dict[int, tuple[int, list[tuple[int, int, int]], str]] = {}
    pos = start
    timestamp_previous: int | None = None
    record_previous: int | None = None
    monotonic = True
    record_count = 0
    n_power = 0
    n_positive = 0
    sum_power = 0
    sum_positive = 0
    timer_started: int | None = None
    timer_seconds = 0
    timer_intervals = 0
    session_timer: float | None = None
    session_power: int | None = None
    while pos < end:
        header = fit[pos]
        pos += 1
        compressed = bool(header & 0x80)
        local = (header >> 5) & 3 if compressed else header & 15
        if not compressed and (header & 0x40):
            if pos + 5 > end:
                raise ValueError("Truncated FIT definition")
            endian = "little" if fit[pos + 1] == 0 else "big"
            message_type = int.from_bytes(fit[pos + 2:pos + 4], endian)
            count = fit[pos + 4]
            pos += 5
            specs = []
            for _ in range(count):
                if pos + 3 > end:
                    raise ValueError("Truncated FIT field definition")
                specs.append(tuple(fit[pos:pos + 3]))
                pos += 3
            if header & 0x20:
                if pos >= end:
                    raise ValueError("Truncated developer field count")
                dev_count = fit[pos]
                pos += 1 + dev_count * 3
                if pos > end:
                    raise ValueError("Truncated developer field definitions")
            definitions[local] = (message_type, specs, endian)
            continue
        if local not in definitions:
            raise ValueError("Unrecognized local FIT message")
        message_type, specs, endian = definitions[local]
        fields: dict[int, int] = {}
        for field_no, size, base in specs:
            if pos + size > end:
                raise ValueError("FIT field exceeds data section")
            if message_type in (18, 20, 21) and field_no in (0, 1, 7, 8, 20, 253):
                value = _integer(fit[pos:pos + size], base, endian)
                if value is not None:
                    fields[field_no] = value
            pos += size
        stamp = fields.get(253)
        if stamp is None and compressed and timestamp_previous is not None:
            offset = header & 31
            stamp = (timestamp_previous & ~31) | offset
            if stamp < timestamp_previous:
                stamp += 32
        if stamp is not None:
            timestamp_previous = stamp
        if message_type == 21 and fields.get(0) == 0 and stamp is not None:
            typ = fields.get(1)
            if typ == 0 and timer_started is None:
                timer_started = stamp
            elif typ in (1, 4, 8, 9) and timer_started is not None:
                timer_seconds += max(0, stamp - timer_started)
                timer_intervals += 1
                timer_started = None
        elif message_type == 20:
            record_count += 1
            if stamp is not None:
                if record_previous is not None and stamp < record_previous:
                    monotonic = False
                record_previous = stamp
            if 7 in fields:
                power = fields[7]
                n_power += 1
                sum_power += power
                if power > 0:
                    n_positive += 1
                    sum_positive += power
        elif message_type == 18:
            if 8 in fields:
                session_timer = fields[8] / 1000
            if 20 in fields:
                session_power = fields[20]
    if not record_count or session_timer is None:
        raise ValueError("Missing FIT session timer or records")
    delta = round(timer_seconds - session_timer, 3)
    timer_ok = timer_started is None and monotonic and abs(delta) <= max(2.0, session_timer * 0.001)
    inclusive = sum_power / n_power if n_power else None
    exclusive = sum_positive / n_positive if n_positive else None
    tolerance = max(3.0, 0.03 * session_power) if session_power is not None else None
    if session_power is None or not n_power:
        match = "NO_GARMIN_POWER"
    elif inclusive is not None and abs(inclusive - session_power) <= tolerance:
        match = "ZERO_INCLUSIVE_MATCH"
    elif exclusive is not None and abs(exclusive - session_power) <= tolerance:
        match = "ZERO_EXCLUDED_MATCH"
    else:
        match = "POWER_DIFF_REVIEW"
    return {
        "activity_id": activity_id,
        "fit_crc": "PASS",
        "record_count": record_count,
        "timer_intervals": timer_intervals,
        "timer_session_seconds": round(session_timer, 3),
        "timer_reconstructed_seconds": timer_seconds,
        "timer_difference_seconds": delta,
        "timebase_status": "TIMER_VERIFIED" if timer_ok else "TIMER_REVIEW",
        "power_samples": n_power,
        "zero_power_samples": n_power - n_positive,
        "fit_avg_power_with_zeros_w": round(inclusive, 2) if inclusive is not None else None,
        "fit_avg_power_without_zeros_w": round(exclusive, 2) if exclusive is not None else None,
        "garmin_session_avg_power_w": session_power,
        "power_policy_match": match,
    }
