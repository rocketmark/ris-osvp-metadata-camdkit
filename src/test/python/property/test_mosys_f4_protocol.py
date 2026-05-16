#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the SMTPE RIS OSVP Metadata Project

"""Property-based tests for the Mo-Sys F4 binary protocol parser.

Generates synthetic F4 binary streams and verifies round-trip properties
of the parser. Covers the frame-count contract, axis value encoding/decoding,
timecode extraction, checksum rejection, and frame number extraction.

Structure:
  Section 1 — F4PacketBuilder: synthesizes valid F4 binary from parameters.
  Section 2 — Hypothesis strategies for each axis type.
  Section 3 — Property tests using F4PacketBuilder + strategies.
"""

import os
import struct
import tempfile
import unittest

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from camdkit.mosys import reader
from camdkit.mosys.f4 import F4, F4PacketParser


# ---------------------------------------------------------------------------
# Section 1 — F4 Packet Builder
# ---------------------------------------------------------------------------

def _compute_checksum(packet_without_checksum: bytes) -> int:
    """Compute the F4 checksum: (0x40 - sum(all preceding bytes)) % 256."""
    total = 0x40
    for b in packet_without_checksum:
        total -= b
    return total % 256


def build_axis_block(axis_id: int, axis_status: int, d1: int, d2: int, d3: int) -> bytes:
    return bytes([axis_id & 0xFF, axis_status & 0xFF, d1 & 0xFF, d2 & 0xFF, d3 & 0xFF])


def build_packet(camera_id: int = 1, status: int = 0,
                 axis_blocks: list[bytes] | None = None) -> bytes:
    """Build a complete valid F4 packet with correct checksum.

    Includes _required_axis_blocks() by default. Pass axis_blocks explicitly
    to add test-specific axes; required axes are always prepended.
    """
    extra = axis_blocks or []
    all_blocks = _required_axis_blocks() + extra
    axis_count = len(all_blocks)
    header = bytes([F4.COMMAND_BYTE, camera_id & 0xFF, axis_count & 0xFF, status & 0xFF])
    body = b''.join(all_blocks)
    packet = header + body
    return packet + bytes([_compute_checksum(packet)])


def build_stream(packets: list[bytes]) -> bytes:
    """Concatenate packets into a byte stream suitable for to_clip()."""
    return b''.join(packets)


def _required_axis_blocks() -> list[bytes]:
    """Axis blocks required to prevent parser crashes in get_tracking_frame().

    Three unconditional constructions in the parser crash without these:
    1. Synchronization(frequency=None) if no timecode axis is present.
    2. 36.0 / (2 * tan(fov_h/2)) ZeroDivisionError if no focal length axis
       is present (fov_h defaults to 0.0).
    3. FizEncoders(None, None, None) ValueError if no encoder axis is present.
    """
    return [
        timecode_axis_block(hours=0, minutes=0, seconds=0, frames=0, fps_bits=0b01),
        float_axis_block(F4.FIELD_ID_FOCAL_LENGTH_FX, 45.0),
        encoder_axis_block(F4.FIELD_ID_FOCUS, 32768),  # 50% focus
    ]


# --- Typed axis block helpers ---

def angle_axis_block(axis_id: int, millidegrees: int) -> bytes:
    """Encode a pan/tilt/roll/x/y/z value as a signed 24-bit integer."""
    val = millidegrees & 0xFFFFFF
    d1 = (val >> 16) & 0xFF
    d2 = (val >> 8) & 0xFF
    d3 = val & 0xFF
    return build_axis_block(axis_id, 0, d1, d2, d3)


def encoder_axis_block(axis_id: int, raw: int) -> bytes:
    """Encode a normalized encoder value as a 16-bit integer in d2/d3."""
    raw = raw & 0xFFFF
    d2 = (raw >> 8) & 0xFF
    d3 = raw & 0xFF
    return build_axis_block(axis_id, 0, 0, d2, d3)


def timecode_axis_block(hours: int, minutes: int, seconds: int,
                        frames: int, fps_bits: int) -> bytes:
    """
    Encode a SMPTE timecode into the F4 timecode axis block (axis_id=0xF8).

    Bit layout (from F4AxisBlock.to_timecode):
      axis_status bits 6:5 = fps_bits (00=24, 01=25, 10=30)
      d1 = [h5..h0 | m5 m4]
      d2 = [m3..m0 | s5..s2]
      d3 = [s1 s0 | f5..f0]
    """
    axis_status = (fps_bits & 0x3) << 5
    d1 = ((hours & 0x3F) << 2) | ((minutes >> 4) & 0x3)
    d2 = ((minutes & 0xF) << 4) | ((seconds >> 2) & 0xF)
    d3 = ((seconds & 0x3) << 6) | (frames & 0x3F)
    return build_axis_block(F4.FIELD_ID_TIMECODE, axis_status, d1, d2, d3)


def status_axis_block(status_index: int) -> bytes:
    """Encode a tracking status string index into d2 bits 7:4."""
    d2 = (status_index & 0xF) << 4
    return build_axis_block(F4.TRACKING_STATUS, 0, 0, d2, 0)


def float_axis_block(axis_id: int, value: float) -> bytes:
    """Encode an IEEE 754 float across axis_status + d1 + d2 + d3."""
    packed = struct.pack('!f', value)
    return build_axis_block(axis_id, packed[0], packed[1], packed[2], packed[3])


def _to_clip_from_bytes(data: bytes, frames: int = -1):
    """Write bytes to a temp file and call to_clip() — mirrors reader interface."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.f4') as f:
        f.write(data)
        tmp_path = f.name
    try:
        return reader.to_clip(tmp_path, frames)
    finally:
        os.unlink(tmp_path)


def _parse_single_packet(packet: bytes):
    """Parse one F4 packet via F4PacketParser. Returns (success, frame_clip)."""
    parser = F4PacketParser()
    success = parser.initialise(packet)
    if success:
        return True, parser.get_tracking_frame()
    return False, None


# ---------------------------------------------------------------------------
# Section 2 — Hypothesis strategies
# ---------------------------------------------------------------------------

# 24-bit signed integer range for angle/linear axes
signed_24bit = st.integers(min_value=-8_388_608, max_value=8_388_607)

# 16-bit unsigned range for normalized encoder axes
unsigned_16bit = st.integers(min_value=0, max_value=65535)

# Number of frames to request from to_clip()
frame_count = st.integers(min_value=1, max_value=30)

# Valid timecode field ranges for 24fps (most restrictive frame count)
valid_timecode = st.fixed_dictionaries({
    "hours":   st.integers(0, 23),
    "minutes": st.integers(0, 59),
    "seconds": st.integers(0, 59),
    "frames":  st.integers(0, 23),
    "fps_bits": st.sampled_from([0b00, 0b01, 0b10]),  # 24/25/30
})

# Valid tracking status indices
status_index = st.integers(min_value=0, max_value=len(F4.TRACKING_STATUS_STRINGS) - 1)

# Packet status byte: frame_number in bits 3:0, recording in bit 4, sync in bit 5
packet_status = st.integers(min_value=0, max_value=0x3F)


# ---------------------------------------------------------------------------
# Section 3 — Property tests
# ---------------------------------------------------------------------------

class F4FrameCountTests(unittest.TestCase):

    @given(frame_count)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_frame_count_contract(self, n):
        """to_clip(path, n) must return exactly n frames — not n+1.

        Regression property for the off-by-one fixed in SMPTE PR #206
        (count <= frames → count < frames). A Hypothesis-driven synthetic
        stream catches this class of error without needing the real corpus file.
        """
        packets = [
            build_packet(camera_id=1, status=i % 16,
                         axis_blocks=[angle_axis_block(F4.FIELD_ID_PAN, 0)])
            for i in range(n + 5)  # stream longer than requested
        ]
        clip = _to_clip_from_bytes(build_stream(packets), frames=n)
        self.assertEqual(len(clip.sample_id), n)


class F4AngleRoundtripTests(unittest.TestCase):

    @given(signed_24bit)
    def test_pan_roundtrip(self, millidegrees):
        """Pan value survives encode → parse within floating-point tolerance."""
        packet = build_packet(axis_blocks=[angle_axis_block(F4.FIELD_ID_PAN, millidegrees)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        expected = millidegrees / 1000.0
        self.assertAlmostEqual(frame.transforms[0][0].rotation.pan, expected, places=3)

    @given(signed_24bit)
    def test_tilt_roundtrip(self, millidegrees):
        """Tilt value survives encode → parse within floating-point tolerance."""
        packet = build_packet(axis_blocks=[angle_axis_block(F4.FIELD_ID_TILT, millidegrees)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        expected = millidegrees / 1000.0
        self.assertAlmostEqual(frame.transforms[0][0].rotation.tilt, expected, places=3)

    @given(signed_24bit)
    def test_roll_roundtrip(self, millidegrees):
        """Roll value survives encode → parse within floating-point tolerance."""
        packet = build_packet(axis_blocks=[angle_axis_block(F4.FIELD_ID_ROLL, millidegrees)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        expected = millidegrees / 1000.0
        self.assertAlmostEqual(frame.transforms[0][0].rotation.roll, expected, places=3)

    @given(signed_24bit, signed_24bit, signed_24bit)
    def test_translation_xyz_roundtrip(self, x_mm, y_mm, z_mm):
        """X/Y/Z translation values survive encode → parse within floating-point tolerance."""
        packet = build_packet(axis_blocks=[
            angle_axis_block(F4.FIELD_ID_X, x_mm),
            angle_axis_block(F4.FIELD_ID_Y, y_mm),
            angle_axis_block(F4.FIELD_ID_HEIGHT, z_mm),
        ])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        t = frame.transforms[0][0].translation
        self.assertAlmostEqual(t.x, x_mm / 1000.0, places=3)
        self.assertAlmostEqual(t.y, y_mm / 1000.0, places=3)
        self.assertAlmostEqual(t.z, z_mm / 1000.0, places=3)


class F4EncoderRoundtripTests(unittest.TestCase):

    @given(unsigned_16bit)
    def test_focus_encoder_roundtrip(self, raw):
        """Focus encoder raw value [0..65535] decodes to raw/65536."""
        packet = build_packet(axis_blocks=[encoder_axis_block(F4.FIELD_ID_FOCUS, raw)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        self.assertAlmostEqual(frame.lens_encoders[0].focus, raw / 65536.0, places=6)

    @given(unsigned_16bit)
    def test_zoom_encoder_roundtrip(self, raw):
        """Zoom encoder raw value [0..65535] decodes to raw/65536."""
        packet = build_packet(axis_blocks=[encoder_axis_block(F4.FIELD_ID_ZOOM, raw)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        self.assertAlmostEqual(frame.lens_encoders[0].zoom, raw / 65536.0, places=6)

    @given(unsigned_16bit)
    def test_iris_encoder_roundtrip(self, raw):
        """Iris encoder raw value [0..65535] decodes to raw/65536."""
        packet = build_packet(axis_blocks=[encoder_axis_block(F4.FIELD_ID_IRIS, raw)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        self.assertAlmostEqual(frame.lens_encoders[0].iris, raw / 65536.0, places=6)


class F4TimecodeTests(unittest.TestCase):

    @given(valid_timecode)
    def test_timecode_roundtrip(self, tc):
        """Timecode fields survive the F4 bit-packing encoding → parse."""
        block = timecode_axis_block(tc["hours"], tc["minutes"], tc["seconds"],
                                    tc["frames"], tc["fps_bits"])
        packet = build_packet(axis_blocks=[block])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        parsed = frame.timing_timecode[0]
        self.assertEqual(parsed.hours,   tc["hours"])
        self.assertEqual(parsed.minutes, tc["minutes"])
        self.assertEqual(parsed.seconds, tc["seconds"])
        self.assertEqual(parsed.frames,  tc["frames"])


class F4ChecksumTests(unittest.TestCase):

    @given(signed_24bit)
    def test_valid_packet_accepted(self, millidegrees):
        """Any well-formed packet with correct checksum is accepted."""
        packet = build_packet(axis_blocks=[angle_axis_block(F4.FIELD_ID_PAN, millidegrees)])
        success, _ = _parse_single_packet(packet)
        self.assertTrue(success)

    def test_corrupted_checksum_rejected(self):
        """Flipping the last data byte causes checksum validation to fail."""
        packet = bytearray(
            build_packet(axis_blocks=[angle_axis_block(F4.FIELD_ID_PAN, 1000)])
        )
        # Corrupt the byte just before the checksum (last body byte)
        packet[-2] ^= 0xFF
        success, _ = _parse_single_packet(bytes(packet))
        self.assertFalse(success)


class F4FrameNumberTests(unittest.TestCase):

    @given(packet_status)
    def test_frame_number_extracted_from_status(self, status):
        """timing_sequence_number == status % 16 (bottom 4 bits of status byte)."""
        packet = build_packet(status=status,
                              axis_blocks=[angle_axis_block(F4.FIELD_ID_PAN, 0)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        self.assertEqual(frame.timing_sequence_number[0], status % 16)

    @given(packet_status)
    def test_recording_flag_extracted_from_status(self, status):
        """tracker_recording == bool(status bit 4)."""
        packet = build_packet(status=status,
                              axis_blocks=[angle_axis_block(F4.FIELD_ID_PAN, 0)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        expected = (status & (1 << 4)) != 0
        self.assertEqual(frame.tracker_recording[0], expected)


class F4StatusStringTests(unittest.TestCase):

    @given(status_index)
    def test_tracking_status_string_roundtrip(self, idx):
        """Tracking status index decodes to the correct string from TRACKING_STATUS_STRINGS."""
        packet = build_packet(axis_blocks=[status_axis_block(idx)])
        success, frame = _parse_single_packet(packet)
        self.assertTrue(success)
        self.assertEqual(frame.tracker_status[0], F4.TRACKING_STATUS_STRINGS[idx])


if __name__ == "__main__":
    unittest.main()
