#!/usr/bin/env python3
"""
liveness_test.py — TEC liveness test (test-2) for the 16-sensor board
NAU Cybersecurity Lab

Two sub-commands:

  enroll  Build TT1 from multi-temperature data and save it with the
          normalization stats used to build it.
            python liveness_test.py enroll --data combinedwtemp.csv [--plot]
          -> tt1.csv, norm_stats.json

  test    Read the board at the current temperature and run the liveness test
          against tt1.csv.
            python liveness_test.py test --port /dev/ttyUSB0 --temp 23
            python liveness_test.py test --port COM3 --temp 23 --test 1 --pairs 200
            python liveness_test.py test --port COM3 --temp 23 --reps 100 \
                --pairs-per-cat 40 --seed 7 --raw-out raw.csv --report rep.json

Liveness tests (whitepaper §3.1):
  * Pairs are split by the distance of their exact crossing temperature
    from T (edges set by BANDS or --bands, default 1,5,10):
      at     ≤1°C
      close  1–5°C
      far    5–10°C       (pairs >10°C away are unused)
  * --test 1: one random number picks --pairs pairs from all of TT1; they are
              then sorted into the three populations.
    --test 2: a separate random number per population picks --pairs-per-cat
              pairs from each (default).
  * The board is read --reps times in the zmax orientation. Each pair gives a
    response bit per read: 0 if first cell < second cell, else 1.
  * Error rate = fraction of bits that differ from the TT1 prediction at T.
    Flip rate = per-pair minority fraction across reads (instability).
  * PASS if err(at) > err(close) > err(far) and err(far) <= --far-max.

Needs read_board.py in the same directory.
Exit codes: 0 pass, 1 fail, 2 inconclusive / error.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

AXES = ["x", "y", "z"]
AXIS_IDX = {a: i for i, a in enumerate(AXES)}
T_LOW, T_HIGH = 10, 60            # enrollment endpoints actually measured
BUCKET = 5                        # TT1 row spacing, °C

# Liveness categories: max |crossing_temp - T| in °C for (at, close, far).
# at: d <= 1 | close: 1 < d <= 5 | far: 5 < d <= 10 | beyond 10: unused.
# Override per run with --bands AT,CLOSE,FAR (e.g. --bands 1,5,100).
BANDS = (1.0, 5.0, 10.0)
T_MIN, T_MAX = 0, 70              # TT1 row range, °C
TT1_COLS = ["temperature", "crossing_temp", "sensor1", "axis1",
            "sensor2", "axis2", "dir", "value"]


def to_bucket(t):
    """Round to the nearest TT1 row (half-up, not banker's rounding)."""
    return int(np.floor(t / BUCKET + 0.5)) * BUCKET


# ══════════════════════════════════════════════════════════════════════════════
# Enrollment
# ══════════════════════════════════════════════════════════════════════════════

def load_table(filename):
    df = pd.read_csv(filename)
    return df[["sensor", "x", "y", "z", "temperature"]].copy()


def compute_norm_stats(df):
    """Per-axis mean/std over the whole enrollment set. Temperature untouched."""
    return {a: {"mean": float(df[a].mean()), "std": float(df[a].std())}
            for a in AXES}


def normalize(df, stats):
    df = df.copy()
    for a in AXES:
        df[a] = (df[a] - stats[a]["mean"]) / stats[a]["std"]
    return df


def create_spaghetti_table(df, plot=False):
    """
    Build one line per (sensor, axis) from the T_LOW and T_HIGH readings and
    find every pair of lines that crosses between them.

    Pairs are ordered so the "first cell" is the lower (sensor, axis):
    sensor number first, then x < y < z.

    Returns (lines, intersections DataFrame).
    """
    ep = (df[df["temperature"].isin([T_LOW, T_HIGH])]
          .groupby(["sensor", "temperature"])[AXES].mean()
          .reset_index())
    lo = ep[ep["temperature"] == T_LOW].set_index("sensor")
    hi = ep[ep["temperature"] == T_HIGH].set_index("sensor")

    lines = []
    for s in sorted(lo.index):
        if s not in hi.index:
            continue
        for a in AXES:
            lines.append({"sensor": int(s), "axis": a,
                          "y_lo": float(lo.loc[s, a]),
                          "y_hi": float(hi.loc[s, a])})
    lines.sort(key=lambda l: (l["sensor"], AXIS_IDX[l["axis"]]))

    rows = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i], lines[j]
            d_lo = a["y_lo"] - b["y_lo"]
            d_hi = a["y_hi"] - b["y_hi"]
            if d_lo * d_hi >= 0:
                continue                      # no sign change -> no crossing

            # a(t) - b(t) = d_lo + t (d_hi - d_lo) = 0,  t in (0, 1)
            t = d_lo / (d_lo - d_hi)
            temp = T_LOW + t * (T_HIGH - T_LOW)
            value = a["y_lo"] + t * (a["y_hi"] - a["y_lo"])

            # Whitepaper §2.1.1 step 5: '+' when the first cell rises faster
            # than the second -> state 0 (first < second) below the crossing,
            # 1 above. Since the lines cross, first rising faster <=> first
            # starts below the second, i.e. d_lo < 0.
            direction = "+" if d_lo < 0 else "-"

            rows.append({"sensor1": a["sensor"], "axis1": a["axis"],
                         "sensor2": b["sensor"], "axis2": b["axis"],
                         "crossing_temp": temp, "value": value,
                         "dir": direction})

    intersections = pd.DataFrame(
        rows, columns=["sensor1", "axis1", "sensor2", "axis2",
                       "crossing_temp", "value", "dir"])

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        for l in lines:
            ax.plot([T_LOW, T_HIGH], [l["y_lo"], l["y_hi"]], linewidth=1)
        if not intersections.empty:
            ax.scatter(intersections["crossing_temp"], intersections["value"],
                       s=30, zorder=3)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Normalized value")
        ax.set_title(f"Sensor Axis Changes: {T_LOW}°C → {T_HIGH}°C")
        ax.set_xticks([T_LOW, T_HIGH])
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()

    return lines, intersections


def create_tt1(intersections):
    """
    One row per crossing pair:
      temperature    TT1 row (crossing temp rounded to nearest 5°C)
      crossing_temp  exact interpolated crossing temperature
      sensor1/axis1  first cell
      sensor2/axis2  second cell
      dir            '+' -> state 0 below crossing, 1 above; '-' the reverse
      value          normalized value at the crossing
    """
    if intersections.empty:
        return pd.DataFrame(columns=TT1_COLS)
    tt1 = intersections.copy()
    tt1["temperature"] = tt1["crossing_temp"].apply(to_bucket)
    tt1 = tt1[(tt1["temperature"] >= T_MIN) & (tt1["temperature"] <= T_MAX)]
    tt1 = tt1.sort_values(["temperature", "crossing_temp"]).reset_index(drop=True)
    return tt1[TT1_COLS]


def cmd_enroll(args):
    df = load_table(args.data)
    stats = compute_norm_stats(df)
    df = normalize(df, stats)

    _, intersections = create_spaghetti_table(df, plot=args.plot)
    tt1 = create_tt1(intersections)

    tt1.to_csv(args.tt1, index=False)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    counts = tt1.groupby("temperature").size().reindex(
        range(T_MIN, T_MAX + 1, BUCKET), fill_value=0)
    print(f"\n  {len(tt1)} crossing pairs -> {args.tt1}")
    print(f"  norm stats          -> {args.stats}\n")
    print("  temp  pairs")
    for t, n in counts.items():
        print(f"  {t:>4}  {n:>5}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Liveness test
# ══════════════════════════════════════════════════════════════════════════════

def categorize(tt1, T, bands=BANDS):
    """
    Split pairs by |crossing_temp - T| using the band edges (at, close, far):
      at     d <= at
      close  at < d <= close
      far    close < d <= far        (pairs beyond far are unused)
    """
    at, close, far = bands
    d = (tt1["crossing_temp"] - T).abs()
    return {"at":    tt1[d <= at],
            "close": tt1[(d > at) & (d <= close)],
            "far":   tt1[(d > close) & (d <= far)]}


def random_streams(seed, k):
    """
    k independent random streams from one seed. With seed=None the seed comes
    from OS entropy; it is returned so the run can be reproduced with --seed.
    """
    ss = np.random.SeedSequence(seed)
    return ss.entropy, [np.random.default_rng(c) for c in ss.spawn(k)]


def subsample(df, n, rng):
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.iloc[rng.choice(len(df), n, replace=False)].reset_index(drop=True)


def draw_test1(tt1, T, n, rng, bands):
    """Test-1: one random number picks n pairs from all of TT1; the picked
    pairs are then split into at / close / far relative to T."""
    return categorize(subsample(tt1, n, rng), T, bands)


def draw_test2(cats, n, rngs):
    """Test-2: a separate random number (stream) per population."""
    return {k: subsample(df, n, r) for (k, df), r in zip(cats.items(), rngs)}


def expected_bits(df, T):
    below = T < df["crossing_temp"].to_numpy()
    plus = (df["dir"] == "+").to_numpy()
    # '+': 0 below, 1 above.   '-': 1 below, 0 above.
    return np.where(plus, ~below, below).astype(int)


def response_bits(df, reads):
    """reads: (R, n_sensors, 3) normalized. Returns (R, P) bits."""
    s1 = df["sensor1"].to_numpy(int) - 1
    s2 = df["sensor2"].to_numpy(int) - 1
    a1 = df["axis1"].map(AXIS_IDX).to_numpy(int)
    a2 = df["axis2"].map(AXIS_IDX).to_numpy(int)
    return (reads[:, s1, a1] > reads[:, s2, a2]).astype(int)


def score(bits, expected):
    if bits.shape[1] == 0:
        return float("nan"), float("nan")
    err = float((bits != expected[None, :]).mean())
    ones = bits.mean(axis=0)
    flip = float(np.minimum(ones, 1 - ones).mean())
    return err, flip


def normalize_reads(raw, stats):
    mean = np.array([stats[a]["mean"] for a in AXES])
    std = np.array([stats[a]["std"] for a in AXES])
    return (raw - mean) / std


# ── Board I/O ─────────────────────────────────────────────────────────────────

def zmax_locked(r, rb):
    face, _, _, m = rb.detect_orientation(r)
    return face == "zmax" and m[2] > rb.G_LSB * 0.85, m


def wait_for_zmax(ser, n_sensors, timeout, rb, need=3):
    """Block until `need` consecutive cycles are LOCKED in zmax."""
    deadline = time.time() + timeout
    streak = 0
    while time.time() < deadline:
        ser.reset_input_buffer()
        r = rb.read_cycle(ser, n_sensors)
        if r is None:
            streak = 0
            print(f"\r  {rb.warn('waiting for data...')}{' ' * 50}",
                  end="", flush=True)
            continue
        locked, m = zmax_locked(r, rb)
        streak = streak + 1 if locked else 0
        state = rb.ok("zmax LOCKED") if locked else rb.warn("place board flat, chip face UP")
        print(f"\r  Ax={m[0]:+7.1f}  Ay={m[1]:+7.1f}  Az={m[2]:+7.1f}   {state}{' ' * 10}",
              end="", flush=True)
        if streak >= need:
            print()
            return True
    print()
    return False


def capture(ser, n_sensors, reps, rb, writer=None):
    """Grab `reps` fresh zmax cycles. Off-orientation cycles are discarded."""
    got, attempts, rejected = [], 0, 0
    while len(got) < reps and attempts < reps * 6:
        attempts += 1
        ser.reset_input_buffer()          # fresh data, not the backlog
        time.sleep(0.04)
        r = rb.read_cycle(ser, n_sensors)
        if r is None:
            continue
        locked, _ = zmax_locked(r, rb)
        if not locked:
            rejected += 1
            continue
        got.append(r)
        if writer:
            ts = datetime.now().isoformat(timespec="milliseconds")
            writer.writerows(rb._rows(False, ts, 1, len(got), r, "zmax"))
        print(f"\r  capturing {len(got)}/{reps}   rejected {rejected}   ",
              end="", flush=True)
    print()
    return (np.stack(got) if got else None), rejected


# ── Command ───────────────────────────────────────────────────────────────────

def cmd_test(args):
    import read_board as rb
    try:
        import serial
    except ImportError:
        print(rb.err("\n  [!] pip install pyserial\n"))
        return 2

    tt1 = pd.read_csv(args.tt1)
    with open(args.stats) as f:
        stats = json.load(f)

    T = args.temp
    if not (T_LOW <= T <= T_HIGH):
        print(rb.warn(f"  [!] T={T}°C is outside the enrolled range "
                      f"{T_LOW}–{T_HIGH}°C; predictions are extrapolated."))

    if args.test == 1 and args.pairs_per_cat is not None:
        print(rb.err("  [!] --pairs-per-cat is for --test 2; use --pairs with --test 1\n"))
        return 2
    if args.test == 2 and args.pairs is not None:
        print(rb.err("  [!] --pairs is for --test 1; use --pairs-per-cat with --test 2\n"))
        return 2

    bands = tuple(args.bands)
    if not (0 < bands[0] < bands[1] < bands[2]):
        print(rb.err(f"  [!] --bands must be increasing positive values, got {bands}\n"))
        return 2

    cats = categorize(tt1, T, bands)
    if args.test == 1:
        seed, (rng,) = random_streams(args.seed, 1)
        picked = draw_test1(tt1, T, args.pairs, rng, bands)
        n_drawn = len(tt1) if args.pairs is None else min(args.pairs, len(tt1))
        n_used = sum(len(d) for d in picked.values())
        drawn = (f"{n_drawn}/{len(tt1)} from one random number "
                 f"({n_used} within ±{bands[2]:g}°C used)")
    else:
        seed, rngs = random_streams(args.seed, 3)
        picked = draw_test2(cats, args.pairs_per_cat, rngs)
        drawn = "one random number per category"

    print()
    print(rb.bold(rb.head(f"  TEC liveness test-{args.test}")))
    print(f"  {rb.dim('T         :')} {T}°C")
    print(f"  {rb.dim('bands     :')} at ≤{bands[0]:g}   close ≤{bands[1]:g}   "
          f"far ≤{bands[2]:g} °C from T")
    print(f"  {rb.dim('draw      :')} {drawn}")
    print(f"  {rb.dim('pairs     :')} at {len(picked['at'])}/{len(cats['at'])}   "
          f"close {len(picked['close'])}/{len(cats['close'])}   "
          f"far {len(picked['far'])}/{len(cats['far'])}")
    print(f"  {rb.dim('seed      :')} {seed}  {rb.dim('(pass --seed to reproduce)')}")
    print(f"  {rb.dim('reps      :')} {args.reps}")
    print()

    empty = [k for k, d in picked.items() if d.empty]
    if empty:
        print(rb.err(f"  INCONCLUSIVE — no TT1 pairs in: {', '.join(empty)}\n"))
        return 2

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except Exception as e:
        print(rb.err(f"  [!] Could not open {args.port}: {e}\n"))
        return 2
    time.sleep(0.4)

    fh = writer = None
    if args.raw_out:
        fh = open(args.raw_out, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(rb._header(False, args.n_sensors))

    try:
        if not wait_for_zmax(ser, args.n_sensors, args.lock_timeout, rb):
            print(rb.err("  INCONCLUSIVE — board never locked in zmax\n"))
            return 2
        raw, rejected = capture(ser, args.n_sensors, args.reps, rb, writer)
    except KeyboardInterrupt:
        print(rb.err("\n  aborted\n"))
        return 2
    finally:
        ser.close()
        if fh:
            fh.close()

    if raw is None or len(raw) < args.reps:
        got = 0 if raw is None else len(raw)
        print(rb.err(f"  INCONCLUSIVE — captured {got}/{args.reps} reads "
                     f"({rejected} off-orientation)\n"))
        return 2

    reads = normalize_reads(raw, stats)

    results = {}
    for k in ("at", "close", "far"):
        df = picked[k]
        bits = response_bits(df, reads)
        err, flip = score(bits, expected_bits(df, T))
        results[k] = {"pairs": len(df), "error_rate": err, "flip_rate": flip}

    e = {k: results[k]["error_rate"] for k in results}
    checks = [
        ("err(at) > err(close)", e["at"] > e["close"]),
        ("err(close) > err(far)", e["close"] > e["far"]),
        (f"err(far) <= {args.far_max:.0%}", e["far"] <= args.far_max),
    ]
    passed = all(c for _, c in checks)

    print()
    print(f"  {'category':<9}{'pairs':>7}{'error':>9}{'flip':>9}")
    for k, r in results.items():
        print(f"  {k:<9}{r['pairs']:>7}{r['error_rate']:>9.1%}{r['flip_rate']:>9.1%}")
    print()
    for name, c in checks:
        print(f"  {rb.ok('✓') if c else rb.err('✗')} {name}")
    print()
    print("  " + (rb.bold(rb.ok("LIVENESS: PASS")) if passed
                  else rb.bold(rb.err("LIVENESS: FAIL"))))
    print()

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(),
                       "test": args.test,
                       "temperature": T, "bands": list(bands), "reps": args.reps,
                       "rejected_reads": rejected, "seed": str(seed),
                       "results": results,
                       "checks": {n: bool(c) for n, c in checks},
                       "pass": passed}, f, indent=2)

    return 0 if passed else 1


# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="TEC liveness test")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="build tt1.csv + norm_stats.json")
    e.add_argument("--data", default="combinedwtemp.csv")
    e.add_argument("--tt1", default="tt1.csv")
    e.add_argument("--stats", default="norm_stats.json")
    e.add_argument("--plot", action="store_true", help="show spaghetti plot")

    t = sub.add_parser("test", help="read the board and run the liveness test")
    t.add_argument("--port", required=True)
    t.add_argument("--temp", type=float, required=True,
                   help="current board temperature, °C")
    t.add_argument("--tt1", default="tt1.csv")
    t.add_argument("--stats", default="norm_stats.json")
    t.add_argument("--reps", type=int, default=50)
    t.add_argument("--test", type=int, choices=[1, 2], default=2,
                   help="1 = single random number picks the pair set (test-1); "
                        "2 = three random numbers, one per category (test-2, default)")
    t.add_argument("--pairs", type=int, default=None,
                   help="test-1: pairs drawn from all of TT1 (default: all)")
    t.add_argument("--pairs-per-cat", type=int, default=None,
                   help="test-2: pairs drawn per category (default: all)")
    t.add_argument("--seed", type=int, default=None,
                   help="random seed; printed each run so a run can be reproduced")
    t.add_argument("--far-max", type=float, default=0.05,
                   help="max allowed error rate for far pairs (default 0.05)")
    t.add_argument("--bands", type=lambda s: [float(v) for v in s.split(",")],
                   default=list(BANDS), metavar="AT,CLOSE,FAR",
                   help="max °C from T for at/close/far "
                        f"(default {','.join(f'{b:g}' for b in BANDS)})")
    t.add_argument("--lock-timeout", type=float, default=60,
                   help="seconds to wait for zmax lock")
    t.add_argument("--n_sensors", type=int, default=16)
    t.add_argument("--baud", type=int, default=115200)
    t.add_argument("--raw-out", default=None, help="save raw reads to CSV")
    t.add_argument("--report", default=None, help="save results as JSON")

    args = p.parse_args()
    if args.cmd == "enroll":
        cmd_enroll(args)
        return 0
    return cmd_test(args)


if __name__ == "__main__":
    sys.exit(main())