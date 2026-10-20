"""Export a saved coefficient waveform and its envelopes, not instrument voltages."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.validation import envelope_function, resource_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pulse')
    parser.add_argument('--output', type=Path, default=ROOT/'exports'/'waveform.csv')
    parser.add_argument('--step-us', type=float, default=.001)
    args = parser.parse_args()
    if args.step_us <= 0 or Path(args.pulse).name != args.pulse:
        parser.error('Require a positive step and a pulse name without path components.')
    pr, x, meta = load_pulse(ROOT, args.pulse)
    if meta.get('status') != 'completed':
        raise RuntimeError('Only completed archived pulses may be exported.')
    # A minimum of 12 samples per carrier cycle is used as an export safeguard.
    if args.step_us*max(pr.freq) > 1/12:
        parser.error('Sampling is too sparse: require at least 12 samples per highest carrier cycle.')
    n = int(np.ceil(pr.T/args.step_us))
    if n > 10_000_000:
        parser.error('Requested export is too large; increase the sampling step.')
    t = np.linspace(0., pr.T, n+1)
    a = envelope_function(pr, x)(t)
    u = 2*np.real(np.sum(a*np.exp(-2j*np.pi*t[:, None]*pr.freq), axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(time_us=t, operator_coefficient_MHz=u)).to_csv(args.output, index=False)
    columns = {'time_us': t}
    for j in range(pr.J):
        columns[f'carrier_{j:02d}_real_MHz'] = a[:, j].real
        columns[f'carrier_{j:02d}_imag_MHz'] = a[:, j].imag
    ep = args.output.with_name(args.output.stem+'_envelopes.csv')
    pd.DataFrame(columns).to_csv(ep, index=False)
    sidecar = dict(pulse=args.pulse, config=meta['config'],
        carrier_frequencies_MHz=pr.freq.tolist(), sample_step_us=float(t[1]-t[0]),
        formula='u(t)=2 Re sum_j a_j(t) exp(-2 pi i f_j t); Hdrive/h=u(t) O',
        coefficient_units='MHz', time_units='microseconds', mechanism=pr.mechanism,
        frame=pr.frame, active_branches=list(pr.active), resources=resource_metrics(pr,x),
        warning='Coefficient waveform only. No calibrated STM voltage or hardware feasibility is implied.')
    args.output.with_suffix('.json').write_text(json.dumps(sidecar, indent=2))
    print(args.output, ep, args.output.with_suffix('.json'), sep='\n')


if __name__ == '__main__':
    main()
