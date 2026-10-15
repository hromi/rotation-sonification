#!/usr/bin/env python3
"""
Rotation Sonification — two HoloLens 2 wearers, turning around their own axis.

Collects head pose from HL2SS (Spatial Input, port 3812) on each device,
derives the yaw (rotation around the vertical body axis), unwraps it so whole
turns accumulate, and sonifies it:

  heading        -> Shepard tone chroma. The circle of the body is mapped onto
                    the circle of pitch: turning clockwise rises forever,
                    counter-clockwise falls forever; 1 turn = 1 octave, so there
                    is no seam at 360 degrees.
  heading        -> stereo position (the voice moves as you turn)
  angular speed  -> energy: loudness, spectral brightness, "wind" noise, and a
                    Leslie-like pulse 4x per revolution. Stillness = soft hum.
  passing front  -> a bell. Its pitch walks a pentatonic scale with the winding
                    number: up for clockwise turns, down for counter-clockwise.
  two wearers    -> both start in unison facing the same way; their headings
                    become a living interval, near-alignment is heard as beating.
  ground         -> a quiet A/E drone, the floor the bodies turn upon.

Data is logged to CSV (logs/rotation_<time>.csv) per received packet.

    python rotation_sonify.py                       # the two default HoloLenses
    python rotation_sonify.py --hosts 192.168.1.102
    python rotation_sonify.py --simulate            # no devices needed
    python rotation_sonify.py --simulate --wav demo.wav --duration 30

Keys (type + Enter):  r = recenter (everyone faces "front" now), q = quit.
Note: each HoloLens has its own world frame, so headings are relative to the
moment of (re)centering. Start both wearers facing the same direction.
"""
import argparse
import csv
import math
import os
import sys
import threading
import time
import wave

import numpy as np
from scipy.signal import butter, lfilter

from hl2ss_si import SpatialInputReceiver

TWO_PI = 2.0 * math.pi
PENTATONIC = (0, 2, 4, 7, 9)


# ----------------------------------------------------------------------------
# Data collection
# ----------------------------------------------------------------------------

class CsvLog:
    FIELDS = ("wall_time", "label", "host", "hl_timestamp", "yaw_deg", "turns",
              "omega_deg_s", "fwd_x", "fwd_y", "fwd_z", "pos_x", "pos_y", "pos_z")

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.FIELDS)
        self._lock = threading.Lock()

    def write(self, row):
        with self._lock:
            self._w.writerow(row)

    def flush(self):
        with self._lock:
            self._f.flush()

    def close(self):
        with self._lock:
            self._f.close()


class HoloTracker(threading.Thread):
    """Receives head pose from one HoloLens and maintains unwrapped yaw."""

    STALE_AFTER = 1.0  # seconds without data -> voice fades out

    def __init__(self, label, host, log=None, omega_tau=0.12):
        super().__init__(daemon=True)
        self.label, self.host, self.log = label, host, log
        self.omega_tau = omega_tau
        self.status = "connecting"
        self.generation = 0  # bumped on recenter so the voice can snap
        self._accum = 0.0
        self._offset = 0.0
        self._last_yaw = None
        self._last_ts = None
        self._omega = 0.0
        self._last_seen = 0.0
        self._stop = threading.Event()

    # -- public, read from audio / UI threads --
    @property
    def turns(self):
        return (self._accum - self._offset) / TWO_PI

    @property
    def omega_deg(self):
        return math.degrees(self._omega)

    @property
    def active(self):
        return time.monotonic() - self._last_seen < self.STALE_AFTER

    def recenter(self):
        self._offset = self._accum
        self.generation += 1

    def stop(self):
        self._stop.set()

    # -- receiving --
    def run(self):
        while not self._stop.is_set():
            rx = SpatialInputReceiver(self.host, timeout=3.0)
            try:
                rx.open()
                self.status = "connected"
                while not self._stop.is_set():
                    self._on_packet(rx.get_next_packet())
            except Exception as e:
                self.status = f"reconnecting ({type(e).__name__})"
                time.sleep(1.0)
            finally:
                rx.close()

    def _on_packet(self, pkt):
        si = pkt.payload
        if not si.head_pose_valid:
            return
        f = si.head_pose.forward
        # Looking almost straight up/down: horizontal heading is undefined.
        if math.hypot(f[0], f[2]) < 0.25:
            self._last_seen = time.monotonic()
            return
        # World frame: +y up, initial forward -z. Clockwise (seen from above) = positive.
        yaw = math.atan2(f[0], -f[2])
        if self._last_yaw is None:
            self._last_yaw, self._last_ts = yaw, pkt.timestamp
            self._accum = self._offset = 0.0
        d = (yaw - self._last_yaw + math.pi) % TWO_PI - math.pi
        dt = min(max((pkt.timestamp - self._last_ts) / 1e7, 1e-3), 0.2)  # 100 ns ticks
        self._accum += d
        a = 1.0 - math.exp(-dt / self.omega_tau)
        self._omega += a * (d / dt - self._omega)
        self._last_yaw, self._last_ts = yaw, pkt.timestamp
        self._last_seen = time.monotonic()
        if self.log is not None:
            p = si.head_pose.position
            self.log.write((f"{time.time():.4f}", self.label, self.host, pkt.timestamp,
                            f"{math.degrees(yaw):.3f}", f"{self.turns:.5f}",
                            f"{self.omega_deg:.3f}", f"{f[0]:.5f}", f"{f[1]:.5f}",
                            f"{f[2]:.5f}", f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"))


class SimTracker:
    """Synthetic wearer for testing without hardware."""

    def __init__(self, label, kind):
        self.label, self.host, self.kind = label, "simulated", kind
        self.status, self.generation, self.active = "simulated", 0, True
        self._t0 = time.monotonic()
        self._offset = 0.0
        self.clock = None  # set for offline rendering

    def _now(self):
        return self.clock() if self.clock else time.monotonic() - self._t0

    def _turns_at(self, t):
        if self.kind == 0:  # slow sway, then a few spins, then back
            return 0.15 * math.sin(0.4 * t) + 1.6 * (1 - math.cos(min(t, 40) * math.pi / 20)) * (t > 8)
        return -0.06 * t + 0.1 * math.sin(1.3 * t)  # steady slow counter-clockwise drift

    @property
    def turns(self):
        return self._turns_at(self._now()) - self._offset

    @property
    def omega_deg(self):
        t = self._now()
        return (self._turns_at(t + 0.01) - self._turns_at(t)) * 36000.0

    def recenter(self):
        self._offset = self._turns_at(self._now())
        self.generation += 1

    def start(self):
        pass

    def stop(self):
        pass


# ----------------------------------------------------------------------------
# Sound
# ----------------------------------------------------------------------------

class Bell:
    RATIOS = np.array([1.0, 2.01, 2.76, 4.07, 5.43])
    DECAYS = np.array([3.5, 2.2, 1.5, 0.9, 0.5])
    AMPS = np.array([1.0, 0.5, 0.35, 0.2, 0.12]) / 2.17

    def __init__(self, freq, gain):
        self.freq, self.gain, self.age = freq, gain, 0

    def render(self, n, sr):
        t = (self.age + np.arange(n)) / sr
        self.age += n
        env = self.AMPS[:, None] * np.exp(-t[None, :] / self.DECAYS[:, None])
        return self.gain * np.sum(env * np.sin(TWO_PI * self.freq * self.RATIOS[:, None] * t[None, :]), axis=0)

    def done(self, sr):
        return self.age > sr * 12


class RotationVoice:
    """One wearer: Shepard tone + wind + bells, steered by unwrapped yaw."""

    N = 8  # Shepard partials, octaves apart

    def __init__(self, tracker, sr, f0, pan_bias, bell_root, octaves_per_turn, seed):
        self.tr, self.sr = tracker, sr
        self.f0, self.pan_bias, self.bell_root = f0, pan_bias, bell_root
        self.opt = octaves_per_turn
        self.k = np.arange(self.N, dtype=np.float64)[:, None]
        self.phases = np.zeros((self.N, 1))
        self.turns_s = None
        self.env = 0.0
        self.presence = 0.0
        self.gen = tracker.generation
        self.last_bell = 0
        self.bells = []
        self.rng = np.random.default_rng(seed)
        self.wind_zi = np.zeros(2)

    def _ring(self, boundary):
        semis = PENTATONIC[boundary % 5] + 12 * (boundary // 5)
        semis = (semis + 12) % 36 - 12  # keep within three octaves
        self.bells.append(Bell(self.bell_root * 2 ** (semis / 12), 0.22))

    def render(self, n):
        sr = self.sr
        target = self.tr.turns
        if self.turns_s is None or self.gen != self.tr.generation:
            self.turns_s, self.gen, self.last_bell = target, self.tr.generation, 0
        # glide toward the latest tracker value (packets arrive at ~30-60 Hz)
        t_prev = self.turns_s
        t_new = t_prev + (1 - math.exp(-n / sr / 0.04)) * (target - t_prev)
        self.turns_s = t_new
        ramp = np.arange(1, n + 1) / n
        turns = t_prev + (t_new - t_prev) * ramp
        yaw = TWO_PI * turns

        # energy envelope from angular speed: fast attack, slow settling release
        active = self.tr.active
        e_target = (1 - math.exp(-abs(self.tr.omega_deg) / 120.0)) if active else 0.0
        tau = 0.08 if e_target > self.env else 1.2
        e_prev = self.env
        self.env += (1 - math.exp(-n / sr / tau)) * (e_target - self.env)
        env = e_prev + (self.env - e_prev) * ramp
        p_prev = self.presence
        self.presence += (1 - math.exp(-n / sr / 0.5)) * ((1.0 if active else 0.0) - self.presence)
        presence = p_prev + (self.presence - p_prev) * ramp

        # Shepard tone: partials slide along a circular log-frequency axis
        x = (self.k + turns[None, :] * self.opt) % self.N
        ph = self.phases + TWO_PI * np.cumsum(self.f0 * 2.0 ** x, axis=1) / sr
        self.phases = (ph[:, -1:] % TWO_PI)
        tilt = -0.35 + 0.7 * env[None, :]  # dark when still, brighter when spinning
        window = (0.5 - 0.5 * np.cos(TWO_PI * x / self.N)) ** 2 * 2.0 ** (tilt * (x - self.N / 2))
        tone = np.sum(window * np.sin(ph), axis=0) / 2.2
        leslie = 1 - 0.55 * env * 0.5 * (1 - np.cos(4 * yaw))
        tone *= (0.16 + 0.84 * env) * leslie * presence

        # wind: band-passed noise, the air displaced by turning
        e = self.env
        fc = 220 * 2 ** (4.2 * e)
        b, a = butter(1, [fc / 1.5 / (sr / 2), min(fc * 1.5 / (sr / 2), 0.95)], btype="band")
        wind, self.wind_zi = lfilter(b, a, self.rng.standard_normal(n), zi=self.wind_zi)
        tone += wind * 0.45 * env ** 1.5 * presence

        # bells on passing "front"; hysteresis: same boundary never rings twice in a row
        f_prev, f_new = math.floor(t_prev), math.floor(t_new)
        if f_prev != f_new:
            boundary = max(f_prev, f_new)
            if boundary != self.last_bell:
                self.last_bell = boundary
                self._ring(boundary)
        bell = np.zeros(n)
        for bl in self.bells:
            bell += bl.render(n, sr)
        self.bells = [bl for bl in self.bells if not bl.done(sr)]

        pan = np.clip(0.5 + self.pan_bias + 0.35 * np.sin(yaw), 0, 1)
        sig = tone + bell
        return sig * np.cos(pan * math.pi / 2), sig * np.sin(pan * math.pi / 2), sig


class GroundDrone:
    """Quiet A1/E2/A2 drone with slow breathing — the tonal floor."""

    def __init__(self, sr, level):
        self.sr, self.level = sr, level
        self.f = np.array([55.0, 82.5, 110.0])[:, None]
        self.a = np.array([1.0, 0.45, 0.3])[:, None]
        self.n0 = 0

    def render(self, n):
        t = (self.n0 + np.arange(n)) / self.sr
        self.n0 += n
        breath = 0.75 + 0.25 * np.sin(TWO_PI * 0.05 * t)
        return self.level * breath * np.sum(self.a * np.sin(TWO_PI * self.f * t), axis=0) / 1.75


class BlockFDN:
    """4-line feedback delay network; every delay > block size, so it is vectorisable per block."""

    def __init__(self, sr, block, t60=3.2, damp=0.35):
        self.delays = [d for d in (1499, 1789, 2111, 2503)]
        assert min(self.delays) >= block
        self.bufs = [np.zeros(d) for d in self.delays]
        self.pos = [0] * 4
        self.g = np.array([10 ** (-3 * d / sr / t60) for d in self.delays])
        self.H = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2.0
        self.damp = damp
        self.zi = [np.zeros(1) for _ in range(4)]

    def _read(self, i, n):
        buf, p = self.bufs[i], self.pos[i]
        idx = (p + np.arange(n)) % len(buf)
        return buf[idx]

    def _write(self, i, x):
        buf, p = self.bufs[i], self.pos[i]
        idx = (p + np.arange(len(x))) % len(buf)
        buf[idx] = x
        self.pos[i] = (p + len(x)) % len(buf)

    def process(self, x):
        n = len(x)
        outs = np.array([self._read(i, n) for i in range(4)])
        for i in range(4):
            outs[i], self.zi[i] = lfilter([1 - self.damp], [1, -self.damp], outs[i], zi=self.zi[i])
        fb = self.H @ (outs * self.g[:, None])
        for i in range(4):
            self._write(i, x + fb[i])
        return outs[0] + outs[2], outs[1] + outs[3]


class Engine:
    def __init__(self, trackers, sr=48000, block=512, octaves_per_turn=1.0, reverb=0.32, gain=0.8):
        self.sr, self.block, self.reverb, self.gain = sr, block, reverb, gain
        # Voice A: upper register, bells on A; voice B: an octave lower, bells on E.
        specs = [(27.5, -0.15, 440.0), (13.75, 0.15, 329.63)]
        self.voices = [RotationVoice(tr, sr, f0, pb, br, octaves_per_turn, seed=i)
                       for i, (tr, (f0, pb, br)) in enumerate(zip(trackers, specs))]
        self.drone = GroundDrone(sr, 0.05)
        self.fdn = BlockFDN(sr, block)

    def render(self, n):
        L = np.zeros(n); R = np.zeros(n); send = np.zeros(n)
        for v in self.voices:
            l, r, s = v.render(n)
            L += l; R += r; send += s
        d = self.drone.render(n)
        L += d; R += d
        rl, rr = self.fdn.process(send * 0.5)
        L += self.reverb * rl; R += self.reverb * rr
        return np.tanh(self.gain * np.stack([L, R], axis=1)).astype(np.float32)

    def callback(self, outdata, frames, time_info, status):
        try:
            outdata[:] = self.render(frames)
        except Exception as e:  # never kill the audio stream
            outdata.fill(0)
            print(f"\naudio error: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def status_lines(trackers):
    lines = []
    for tr in trackers:
        t = tr.turns
        heading = (t % 1.0) * 360.0
        w = tr.omega_deg
        bar = "#" * min(30, int(abs(w) / 12))
        lines.append(f"{tr.label} {tr.host:>15}  {tr.status:<22} "
                     f"heading {heading:6.1f}°  turns {t:+7.2f}  ω {w:+7.1f}°/s  {bar}")
    return lines


def run_live(args, trackers, log):
    import sounddevice as sd
    engine = Engine(trackers, sr=args.sr, block=args.block,
                    octaves_per_turn=args.octaves_per_turn, reverb=args.reverb, gain=args.gain)
    quit_ev = threading.Event()

    def keys():
        for line in sys.stdin:
            c = line.strip().lower()
            if c == "r":
                for tr in trackers:
                    tr.recenter()
            elif c == "q":
                break
        quit_ev.set()

    threading.Thread(target=keys, daemon=True).start()
    print("Rotation Sonification — 'r'+Enter recenter, 'q'+Enter quit"
          + (f"   logging -> {log.path}" if log else ""))
    with sd.OutputStream(samplerate=args.sr, blocksize=args.block, channels=2,
                         dtype="float32", device=args.device, callback=engine.callback):
        first = True
        last_flush = time.monotonic()
        try:
            while not quit_ev.is_set():
                lines = status_lines(trackers)
                if not first:
                    sys.stdout.write(f"\x1b[{len(lines)}A")
                sys.stdout.write("".join(f"\x1b[2K{l}\n" for l in lines))
                sys.stdout.flush()
                first = False
                if log and time.monotonic() - last_flush > 1.0:
                    log.flush(); last_flush = time.monotonic()
                quit_ev.wait(0.2)
        except KeyboardInterrupt:
            pass


def run_offline(args, trackers):
    """Render simulated wearers to a WAV file (for auditioning / testing)."""
    engine = Engine(trackers, sr=args.sr, block=args.block,
                    octaves_per_turn=args.octaves_per_turn, reverb=args.reverb, gain=args.gain)
    n_blocks = int(args.duration * args.sr / args.block)
    clock = {"t": 0.0}
    for tr in trackers:
        tr.clock = lambda: clock["t"]
    out = []
    t0 = time.perf_counter()
    for i in range(n_blocks):
        clock["t"] = i * args.block / args.sr
        out.append(engine.render(args.block))
    cpu = time.perf_counter() - t0
    pcm = (np.concatenate(out) * 32767).astype(np.int16)
    with wave.open(args.wav, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(args.sr)
        w.writeframes(pcm.tobytes())
    print(f"wrote {args.wav}: {args.duration:.0f}s audio rendered in {cpu:.2f}s CPU "
          f"({100 * cpu / args.duration:.1f}% real-time load)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="+", default=["192.168.1.102", "192.168.1.150"],
                    help="one or two HoloLens IPs (HL2SS must be running)")
    ap.add_argument("--simulate", action="store_true", help="synthetic wearers instead of HoloLenses")
    ap.add_argument("--wav", help="with --simulate: render offline to this WAV instead of playing")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--octaves-per-turn", type=float, default=1.0,
                    help="Shepard chroma travelled per full body turn (default 1)")
    ap.add_argument("--reverb", type=float, default=0.32)
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--device", default=None, help="sounddevice output device")
    ap.add_argument("--log", default=None, help="CSV path (default logs/rotation_<time>.csv)")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    labels = ["A", "B"]
    log = None
    if args.simulate:
        trackers = [SimTracker("A", 0), SimTracker("B", 1)]
    else:
        if not args.no_log:
            here = os.path.dirname(os.path.abspath(__file__))
            path = args.log or os.path.join(here, "logs", time.strftime("rotation_%Y%m%d_%H%M%S.csv"))
            log = CsvLog(path)
        trackers = [HoloTracker(labels[i], h, log) for i, h in enumerate(args.hosts[:2])]
    for tr in trackers:
        tr.start()

    try:
        if args.simulate and args.wav:
            run_offline(args, trackers)
        else:
            run_live(args, trackers, log)
    finally:
        for tr in trackers:
            tr.stop()
        if log:
            log.close()
            print(f"data saved to {log.path}")


if __name__ == "__main__":
    main()
