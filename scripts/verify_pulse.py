"""Fresh independent propagation of an archived pulse, without reoptimization.

Writes to verification/ by default, preserving the immutable archived results.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import integrate_extended, full_lab_midpoint
from qudit_control.virtual_electron import integrate_second_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pulse', help='Result name without .json or .npz')
    parser.add_argument('--full-step-us', type=float, default=0.00005)
    parser.add_argument('--projected-step-us', type=float, default=0.004)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'verification')
    parser.add_argument('--start-time-us', type=float, default=0.,
                        help='Laboratory start time t0 of the pulse with continuously running carriers')
    args = parser.parse_args()
    if args.start_time_us < 0:
        parser.error('The start time must be non-negative.')
    if min(args.full_step_us, args.projected_step_us) <= 0:
        parser.error('Integration steps must be positive.')
    if Path(args.pulse).name != args.pulse:
        parser.error('Supply a pulse name, not a path.')
    pr, x, meta = load_pulse(ROOT, args.pulse)
    if meta.get('status') != 'completed':
        raise RuntimeError('The archived pulse is not marked completed.')
    started = time.time()
    t0 = args.start_time_us
    projected, _ = integrate_extended(pr, x, scenario=(pr.B, pr.p, 1., t0), max_step=args.projected_step_us)
    second_order = None
    if pr.mechanism != 'nuclear':
        # Electron-coupled drives: the linear projected model is expected to disagree
        # with the full propagation (drive-induced shift); the second-order model is the
        # relevant reduced check.
        second_order = integrate_second_order(pr, x, t0=t0)
    full = full_lab_midpoint(pr, x, dt=args.full_step_us, t0=t0)
    report = dict(pulse=args.pulse, start_time_us=t0, projected_real_linear=projected, second_order=second_order,
                  full_electron=full, carrier_offset_kHz=None if pr.carrier_offset is None else (1e3*pr.carrier_offset).tolist(),
                  elapsed_s=time.time()-started, comparison=None)
    references = [ROOT/'results'/f'validation_{args.pulse}.json',
                  ROOT/'results'/f'robust_validation_{args.pulse}.json']
    for path in references:
        if path.exists() and t0 == 0.:
            old = json.loads(path.read_text())
            old_full = min(old['full_electron'], key=lambda z: z['dt_us'])
            report['comparison'] = dict(archive_file=path.name,
                archived_finest_dt_us=old_full['dt_us'],
                archived_finest_full_fidelity=old_full['worst'],
                fidelity_difference=full['worst']-old_full['worst'])
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '' if t0 == 0. else f'_t0_{t0:g}us'
    destination = args.output_dir / f'{args.pulse}{suffix}_verification.json'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Fresh validation written to {destination}')


if __name__ == '__main__':
    main()
