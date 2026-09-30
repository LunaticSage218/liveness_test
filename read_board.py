#!/usr/bin/env python3
"""
read_board.py — 16-sensor board reader with Enter-triggered capture
NAU Cybersecurity Lab

The live line always shows what the board is doing right now (Ax/Ay/Az and
which orientation you are in). Nothing is written to disk until you press
Enter — that captures N samples and appends them to the CSV.

Controls:
  Enter   capture --samples cycles and write them to the CSV
  q       quit
  Ctrl-C  quit

Usage:
  python read_board.py --port /dev/ttyUSB0 --out dump.csv
  python read_board.py --port /dev/ttyUSB0 --out dump.csv --samples 5
  python read_board.py --port /dev/ttyUSB0 --out dump.csv --wide
  python read_board.py --port /dev/ttyUSB0                      # readout only
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
BAUD      = 115200
N_SENSORS = 16
G_LSB     = 256.0          # LSB per 1 g — change if your board scales differently

# ── ANSI colour ────────────────────────────────────────────────────────────────
_COL = sys.stdout.isatty() or os.environ.get('FORCE_COLOR')
def _c(code, t):
    if not _COL: return t
    C = {'g':'\033[92m','y':'\033[93m','r':'\033[91m','c':'\033[96m',
         'b':'\033[94m','B':'\033[1m','d':'\033[2m','R':'\033[0m'}
    return C.get(code,'') + t + C['R']
ok   = lambda s: _c('g', s)
warn = lambda s: _c('y', s)
err  = lambda s: _c('r', s)
info = lambda s: _c('c', s)
bold = lambda s: _c('B', s)
dim  = lambda s: _c('d', s)
head = lambda s: _c('b', s)

# ── Keyboard (non-blocking, cross-platform) ────────────────────────────────────
if os.name == 'nt':
    import msvcrt

    class raw_mode:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def key_pressed():
        return msvcrt.getwch() if msvcrt.kbhit() else None
else:
    import select
    import termios
    import tty

    class raw_mode:
        """Put the terminal in cbreak so single keys arrive without Enter echo."""
        def __enter__(self):
            self.old = None
            if sys.stdin.isatty():
                self.old = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            return self
        def __exit__(self, *a):
            if self.old is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
            return False

    def key_pressed():
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if r else None

# ── Orientation labels ─────────────────────────────────────────────────────────
_DESC = {
    'xmax': 'on edge  — chip connector side UP',
    'xmin': 'on edge  — chip connector side DOWN',
    'ymax': 'on edge  — +Y side UP',
    'ymin': 'on edge  — -Y side DOWN',
    'zmax': 'flat     — chip face UP',
    'zmin': 'flat     — chip face DOWN (upside-down)',
}

def detect_orientation(r):
    """
    Work out which of the 6 faces the board is closest to right now.
    Returns (face, description, status, mean_xyz).
    face is None when no axis dominates (board is between orientations).
    """
    m = r.mean(axis=0)                       # mean Ax, Ay, Az across all sensors
    axis = int(np.argmax(np.abs(m)))
    val  = float(m[axis])
    face = ('x', 'y', 'z')[axis] + ('max' if val > 0 else 'min')

    if abs(val) > G_LSB * 0.85:
        status = ok('LOCKED')
    elif abs(val) > G_LSB * 0.66:
        status = warn('tilted')
    else:
        return None, 'between orientations', err('unclear'), m
    return face, _DESC[face], status, m

# ── Serial read ────────────────────────────────────────────────────────────────
def read_cycle(ser, n_sensors, timeout=3.0):
    """Collect one full sweep of all sensors. Returns (n_sensors,3) or None."""
    records = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = ser.readline().decode('ascii', errors='ignore').strip()
        except Exception:
            continue
        parts = raw.split(',')
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0]) - 1
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if 0 <= idx < n_sensors:
            records[idx] = (x, y, z)
        if len(records) == n_sensors:
            return np.array([records[i] for i in range(n_sensors)], dtype=float)
    return None

# ── CSV writers ────────────────────────────────────────────────────────────────
def _header(wide, n_sensors):
    if wide:
        cols = ['timestamp', 'capture', 'sample', 'orientation']
        for i in range(1, n_sensors + 1):
            cols += [f's{i}_x', f's{i}_y', f's{i}_z']
        return cols
    return ['timestamp', 'capture', 'sample', 'sensor', 'x', 'y', 'z', 'orientation']

def _rows(wide, ts, capture, sample, r, face):
    tag = face if face else 'unknown'
    if wide:
        flat = []
        for row in r:
            flat += [f'{row[0]:.4f}', f'{row[1]:.4f}', f'{row[2]:.4f}']
        return [[ts, capture, sample, tag] + flat]
    return [[ts, capture, sample, i + 1,
             f'{row[0]:.4f}', f'{row[1]:.4f}', f'{row[2]:.4f}', tag]
            for i, row in enumerate(r)]

# ── One capture ────────────────────────────────────────────────────────────────
def do_capture(ser, n_sensors, samples, capture_no, writer, fh, wide):
    """Grab `samples` fresh cycles and write them. Returns rows written."""
    print()   # keep the live line on screen
    print(f'  {bold(info(f"[ capture {capture_no} ]"))}  {dim(f"{samples} sample(s)")}')

    grabbed, attempt, n_rows = [], 0, 0
    while len(grabbed) < samples and attempt < samples * 6:
        attempt += 1
        ser.reset_input_buffer()      # take data from now, not the backlog
        time.sleep(0.04)
        r = read_cycle(ser, n_sensors)
        if r is None:
            print(f'     {warn(f"timeout on attempt {attempt}")}')
            continue

        s = len(grabbed) + 1
        face, desc, status, m = detect_orientation(r)
        ts = datetime.now().isoformat(timespec='milliseconds')

        if writer:
            rows = _rows(wide, ts, capture_no, s, r, face)
            writer.writerows(rows)
            n_rows += len(rows)

        print(f'     {dim(f"sample {s}/{samples}")}   '
              f'Ax={bold(f"{m[0]:+7.1f}")}  Ay={bold(f"{m[1]:+7.1f}")}  '
              f'Az={bold(f"{m[2]:+7.1f}")}   {bold(face.upper()) if face else dim("--")}')
        grabbed.append(r)

    if not grabbed:
        print(f'     {err("no data captured")}')
        return 0

    avg = np.stack(grabbed).mean(axis=0)
    face, desc, status, m = detect_orientation(avg)
    print(f'     {dim("mean")}         '
          f'Ax={bold(f"{m[0]:+7.1f}")}  Ay={bold(f"{m[1]:+7.1f}")}  '
          f'Az={bold(f"{m[2]:+7.1f}")}   {info(desc)}  {status}')
    if writer:
        fh.flush()
        print(f'     {ok("written")}      {n_rows} rows')
    else:
        print(f'     {dim("not written — no --out given")}')
    print()
    return n_rows

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='Board reader — live orientation, capture on Enter')
    p.add_argument('--port', required=True, metavar='PORT',
                   help='Serial port (e.g. /dev/ttyUSB0 or COM3)')
    p.add_argument('--out', default=None, metavar='FILE',
                   help='CSV output file. Omit to just watch the readout.')
    p.add_argument('--samples', type=int, default=1, metavar='N',
                   help='Cycles recorded per Enter press (default: 1)')
    p.add_argument('--n_sensors', type=int, default=N_SENSORS, metavar='N',
                   help=f'Sensors per sweep (default: {N_SENSORS})')
    p.add_argument('--baud', type=int, default=BAUD, metavar='N')
    p.add_argument('--wide', action='store_true',
                   help='One row per sample (48 sensor columns) instead of one row per sensor')
    p.add_argument('--append', action='store_true',
                   help='Append to an existing CSV instead of overwriting it')
    args = p.parse_args()

    try:
        import serial as _ser
    except ImportError:
        print(err('\n  [!] pip install pyserial\n'))
        sys.exit(1)

    print()
    print(bold(head('  ╔══════════════════════════════════════════════════════════╗')))
    print(bold(head('  ║   Board Reader  --  capture on Enter                     ║')))
    print(bold(head('  ╚══════════════════════════════════════════════════════════╝')))
    print(f'  {dim("port      :")} {bold(args.port)}  {dim(f"@ {args.baud} baud")}')
    print(f'  {dim("sensors   :")} {args.n_sensors}')
    print(f'  {dim("output    :")} {bold(args.out) if args.out else dim("(readout only)")}')
    print(f'  {dim("samples   :")} {args.samples} {dim("per Enter press")}')
    if args.out:
        print(f'  {dim("format    :")} {"wide (1 row/sample)" if args.wide else "long (1 row/sensor)"}'
              f'{dim("  appending") if args.append else ""}')
    print()
    print(f'  {dim("+--  CONTROLS  " + "-"*44 + "+")}')
    print(f'  {dim("|")}  {bold("Enter")}  capture and write     '
          f'{bold("q")} / {bold("Ctrl-C")}  quit          {dim("|")}')
    print(f'  {dim("+" + "-"*58 + "+")}')
    print()

    try:
        ser = _ser.Serial(args.port, args.baud, timeout=2)
    except Exception as e:
        print(err(f'  [!] Could not open {args.port}: {e}\n'))
        sys.exit(1)
    time.sleep(0.4)

    fh = writer = None
    if args.out:
        exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        fh = open(args.out, 'a' if args.append else 'w', newline='')
        writer = csv.writer(fh)
        if not (args.append and exists):
            writer.writerow(_header(args.wide, args.n_sensors))
            fh.flush()

    captures = total_rows = 0

    try:
        with raw_mode():
            while True:
                ser.reset_input_buffer()
                r = read_cycle(ser, args.n_sensors)

                if r is None:
                    print(f'\r  {warn("waiting for data...")}'
                          f'{" "*45}', end='', flush=True)
                else:
                    face, desc, status, m = detect_orientation(r)
                    label = bold(face.upper()) if face else dim('  --  ')
                    print(f'\r  Ax={bold(f"{m[0]:+7.1f}")}  Ay={bold(f"{m[1]:+7.1f}")}  '
                          f'Az={bold(f"{m[2]:+7.1f}")}  {dim("|")}  {label:<14} '
                          f'{info(desc):<44} {status:<10} '
                          f'{dim(f"captures: {captures}")}   ', end='', flush=True)

                k = key_pressed()
                if k is None:
                    continue
                if k in ('\n', '\r'):
                    captures += 1
                    total_rows += do_capture(ser, args.n_sensors, args.samples,
                                             captures, writer, fh, args.wide)
                elif k in ('q', 'Q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        if fh:
            fh.close()

    print(f'\n\n  {dim("captures  :")} {captures}')
    if args.out:
        size = os.path.getsize(args.out) if os.path.exists(args.out) else 0
        print(f'  {ok("OK")} {bold(args.out)}  {dim(f"({total_rows} rows written, {size//1024} KB)")}')
    print()

if __name__ == '__main__':
    main()
