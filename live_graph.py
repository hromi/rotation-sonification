#!/usr/bin/env python3
"""
Live graph of the two wearers' rotation, fed by the CSV that rotation_sonify.py
writes (HL2SS allows one client per port, so we follow the log instead of
connecting a second time).

    python live_graph.py                 # follows the newest logs/rotation_*.csv
    python live_graph.py --csv path.csv --window 90

Left:  compass — current heading (relative to recenter, 0° = front, clockwise),
       with a fading trail of the last seconds.
Right: unwrapped turns over time (each integer line = a bell), and angular speed.
"""
import argparse
import collections
import glob
import math
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"A": "#f5a142", "B": "#3fc1c9"}
BG, FG, GRID = "#111418", "#d8dde3", "#2a3038"


def newest_log():
    files = glob.glob(os.path.join(HERE, "logs", "rotation_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


class CsvFollower:
    """Incrementally reads rows appended to a growing CSV; switches to newer logs."""

    def __init__(self, path, follow_newest):
        self.follow_newest = follow_newest
        self._open(path)

    def _open(self, path):
        self.path = path
        self.f = open(path, "r") if path else None
        self.cols = None
        self.partial = ""

    def rows(self):
        if self.follow_newest:
            p = newest_log()
            if p and p != self.path:
                if self.f:
                    self.f.close()
                self._open(p)
        if not self.f:
            return []
        chunk = self.f.read()
        if not chunk:
            return []
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()  # last piece may be an incomplete line
        out = []
        for line in lines:
            parts = line.strip().split(",")
            if self.cols is None:
                self.cols = {c: i for i, c in enumerate(parts)}
                continue
            if len(parts) != len(self.cols):
                continue
            try:
                out.append((parts[self.cols["label"]],
                            float(parts[self.cols["wall_time"]]),
                            float(parts[self.cols["turns"]]),
                            float(parts[self.cols["omega_deg_s"]]),
                            parts[self.cols["host"]]))
            except ValueError:
                pass
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="log to follow (default: newest, auto-switching)")
    ap.add_argument("--window", type=float, default=60.0, help="seconds of history shown")
    ap.add_argument("--trail", type=float, default=3.0, help="seconds of compass trail")
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    follower = CsvFollower(args.csv or newest_log(), follow_newest=args.csv is None)
    data = {k: collections.deque() for k in COLORS}  # (t, turns, omega)
    hosts = {}

    plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
                         "ytick.color": FG, "axes.edgecolor": GRID, "font.size": 10})
    fig = plt.figure("Rotation Sonification — live", figsize=(13, 6.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], hspace=0.3, wspace=0.18,
                          left=0.04, right=0.98, top=0.9, bottom=0.09)
    ax_c = fig.add_subplot(gs[:, 0], projection="polar", facecolor=BG)
    ax_t = fig.add_subplot(gs[0, 1], facecolor=BG)
    ax_w = fig.add_subplot(gs[1, 1], facecolor=BG, sharex=ax_t)

    ax_c.set_theta_zero_location("N")
    ax_c.set_theta_direction(-1)  # clockwise, as seen from above
    ax_c.set_ylim(0, 1.15)
    ax_c.set_yticks([])
    ax_c.set_xticks([math.radians(a) for a in range(0, 360, 45)])
    ax_c.set_xticklabels(["front", "45°", "90°", "135°", "back", "225°", "270°", "315°"])
    ax_c.grid(color=GRID)
    ax_c.set_title("heading (0° = front at recenter)", pad=14)

    ax_t.set_ylabel("turns (unwrapped)")
    ax_t.grid(color=GRID, alpha=0.6)
    ax_w.set_ylabel("ω  [°/s]")
    ax_w.set_xlabel("seconds ago")
    ax_w.grid(color=GRID, alpha=0.6)
    ax_w.axhline(0, color=GRID, lw=1)

    art = {}
    for k, c in COLORS.items():
        art[k] = dict(
            arrow=ax_c.plot([], [], color=c, lw=3, solid_capstyle="round")[0],
            head=ax_c.plot([], [], "o", color=c, ms=10)[0],
            trail=ax_c.plot([], [], color=c, lw=1.5, alpha=0.45)[0],
            turns=ax_t.plot([], [], color=c, lw=1.6, label=k)[0],
            omega=ax_w.plot([], [], color=c, lw=1.2)[0],
        )
    legend = ax_t.legend(loc="upper left", facecolor=BG, edgecolor=GRID)
    title = fig.suptitle("", color=FG, fontsize=12)

    def update(_):
        for label, t, turns, omega, host in follower.rows():
            if label in data:
                data[label].append((t, turns, omega))
                hosts[label] = host
        now = max((d[-1][0] for d in data.values() if d), default=None)
        if now is None:
            title.set_text(f"waiting for data in {follower.path or 'logs/ (no log yet)'}")
            return
        tmin, tmax = math.inf, -math.inf
        status = []
        for k, d in data.items():
            while d and d[0][0] < now - args.window:
                d.popleft()
            a = art[k]
            if not d:
                continue
            ts = [now - p[0] for p in d]
            tu = [p[1] for p in d]
            a["turns"].set_data(ts, tu)
            a["omega"].set_data(ts, [p[2] for p in d])
            tmin, tmax = min(tmin, min(tu)), max(tmax, max(tu))
            th = (tu[-1] % 1.0) * 2 * math.pi
            stale = now - d[-1][0] > 1.0
            a["arrow"].set_data([th, th], [0, 1.0])
            a["head"].set_data([th], [1.0])
            for key in ("arrow", "head"):
                a[key].set_alpha(0.25 if stale else 1.0)
            trail = [(p[1] % 1.0) * 2 * math.pi for p in d if p[0] >= now - args.trail]
            radii = [0.55 + 0.45 * i / max(1, len(trail) - 1) for i in range(len(trail))]
            a["trail"].set_data(trail, radii)
            status.append(f"{k} {hosts.get(k, '')}: {(tu[-1] % 1) * 360:5.1f}°  "
                          f"{tu[-1]:+.2f} turns  ω {d[-1][2]:+.0f}°/s" + ("  (stale)" if stale else ""))
        ax_t.set_xlim(args.window, 0)
        if tmin <= tmax:
            pad = 0.25
            lo, hi = math.floor(tmin - pad), math.ceil(tmax + pad)
            ax_t.set_ylim(lo, hi)
            ax_t.set_yticks(range(lo, hi + 1) if hi - lo <= 16 else ax_t.get_yticks())
        wmax = max((abs(p[2]) for d in data.values() for p in d), default=90)
        ax_w.set_ylim(-max(90, wmax * 1.1), max(90, wmax * 1.1))
        title.set_text("      ".join(status))

    anim = FuncAnimation(fig, update, interval=1000 / args.fps, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
