"""
Minimal HL2SS Spatial Input client for Living Piano Joy.

This module intentionally does NOT import the upstream Python `hl2ss` package.
It implements only the Spatial Input stream (TCP port 3812), whose payload is
uncompressed binary data. Dependencies: Python stdlib + NumPy.
"""
from dataclasses import dataclass
import socket
import struct
import numpy as np

SPATIAL_INPUT_PORT = 3812
CHUNK_SIZE = 4096
JOINT_COUNT = 26

class SI_HandJointKind:
    Palm=0; Wrist=1
    ThumbMetacarpal=2; ThumbProximal=3; ThumbDistal=4; ThumbTip=5
    IndexMetacarpal=6; IndexProximal=7; IndexIntermediate=8; IndexDistal=9; IndexTip=10
    MiddleMetacarpal=11; MiddleProximal=12; MiddleIntermediate=13; MiddleDistal=14; MiddleTip=15
    RingMetacarpal=16; RingProximal=17; RingIntermediate=18; RingDistal=19; RingTip=20
    LittleMetacarpal=21; LittleProximal=22; LittleIntermediate=23; LittleDistal=24; LittleTip=25
    TOTAL=26

@dataclass
class HeadPose:
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray

@dataclass
class EyeRay:
    origin: np.ndarray
    direction: np.ndarray

@dataclass
class HandPose:
    orientation: np.ndarray
    position: np.ndarray
    radius: np.ndarray
    accuracy: np.ndarray

@dataclass
class SIFrame:
    head_pose: HeadPose
    eye_ray: EyeRay
    hand_left: HandPose
    hand_right: HandPose
    head_pose_valid: bool
    eye_ray_valid: bool
    hand_left_valid: bool
    hand_right_valid: bool

@dataclass
class Packet:
    timestamp: int
    payload: SIFrame

def decode_si(payload: bytes) -> SIFrame:
    if len(payload) < 4:
        raise ValueError("Spatial Input payload is too short")
    valid = struct.unpack_from('<I', payload, 0)[0]
    f = np.frombuffer(payload, dtype=np.float32, offset=4)
    expected = 9 + 6 + (2 * JOINT_COUNT * 9)
    if f.size < expected:
        raise ValueError(f"Spatial Input payload contains {f.size} floats; expected >= {expected}")

    head_pose = HeadPose(f[0:3].copy(), f[3:6].copy(), f[6:9].copy())
    eye_ray = EyeRay(f[9:12].copy(), f[12:15].copy())
    hands = f[15:15 + 2 * JOINT_COUNT * 9].reshape((-1, 9))
    orientation = hands[:, 0:4]
    position = hands[:, 4:7]
    radius = hands[:, 7]
    accuracy = hands[:, 8].view(np.int32)

    left = HandPose(orientation[:JOINT_COUNT].copy(), position[:JOINT_COUNT].copy(), radius[:JOINT_COUNT].copy(), accuracy[:JOINT_COUNT].copy())
    right = HandPose(orientation[JOINT_COUNT:].copy(), position[JOINT_COUNT:].copy(), radius[JOINT_COUNT:].copy(), accuracy[JOINT_COUNT:].copy())
    return SIFrame(head_pose, eye_ray, left, right,
                   bool(valid & 0x01), bool(valid & 0x02), bool(valid & 0x04), bool(valid & 0x08))

class SpatialInputReceiver:
    """Minimal blocking receiver for HL2SS Spatial Input."""
    def __init__(self, host, port=SPATIAL_INPUT_PORT, chunk_size=CHUNK_SIZE, timeout=None):
        self.host=host; self.port=port; self.chunk_size=chunk_size; self.timeout=timeout
        self._sock=None; self._buffer=bytearray()

    def open(self):
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.timeout is not None: s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        self._sock=s

    def close(self):
        if self._sock is not None:
            try: self._sock.close()
            finally: self._sock=None

    def _recv_more(self):
        chunk=self._sock.recv(self.chunk_size)
        if not chunk: raise ConnectionError("HL2SS Spatial Input connection closed")
        self._buffer.extend(chunk)

    def get_next_packet(self):
        if self._sock is None: raise RuntimeError("SpatialInputReceiver is not open")
        while len(self._buffer) < 12: self._recv_more()
        timestamp, payload_size = struct.unpack_from('<QI', self._buffer, 0)
        packet_size = 12 + payload_size
        while len(self._buffer) < packet_size: self._recv_more()
        payload=bytes(self._buffer[12:packet_size])
        del self._buffer[:packet_size]
        return Packet(timestamp, decode_si(payload))

    def __enter__(self): self.open(); return self
    def __exit__(self, exc_type, exc, tb): self.close()
