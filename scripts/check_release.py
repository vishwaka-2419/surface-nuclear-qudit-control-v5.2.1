"""Check the public scientific package and its frozen SHA-256 manifest.

This validates content and integrity, not whether a DOI has been published or
whether the author has chosen a public reuse licence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def content_checks() -> dict:
    required = [
        'README.md', 'requirements-tested.txt', 'CITATION.cff',
        'LICENSE_NOTICE.md', 'pyproject.toml',
        'manuscript/generated/numbers.tex',
        'supplement/generated/composable_table.tex', 'docs/AI_ASSISTANCE.md',
        'docs/CLAIMS_AND_LIMITATIONS.md', 'docs/REPRODUCING.md',
        'results/robust_validation_robust_N_B1.6_dual_refined.json',
        'results/source_uncertainty_intervals.csv',
    ]
    missing = [p for p in required if not (ROOT / p).is_file()]
    if missing:
        raise RuntimeError('Missing required scientific files: ' + ', '.join(missing))
    private = [p for p in ('author_review', 'submission') if (ROOT / p).exists()]
    if private:
        raise RuntimeError('Author-only directories are present in the public tree: ' + ', '.join(private))
    jobs = []
    for pulse in sorted((ROOT / 'results/pulses').glob('*.npz')):
        path = ROOT / 'results' / (pulse.stem + '.json')
        if not path.is_file():
            raise RuntimeError(f'Missing pulse metadata: {path}')
        meta = json.loads(path.read_text())
        if meta.get('status') != 'completed':
            raise RuntimeError(f'Unfinished pulse: {pulse.stem}')
        if 'config' not in meta or 'parameters' not in meta['config']:
            raise RuntimeError(f'Incomplete model configuration: {pulse.stem}')
        with np.load(pulse, allow_pickle=False) as data:
            if not np.isfinite(data['x']).all():
                raise RuntimeError(f'Nonfinite coefficients: {pulse.stem}')
        jobs.append(pulse.stem)
    if not jobs:
        raise RuntimeError('No archived pulse coefficients found.')
    figures = sorted((ROOT / 'figures').glob('*.pdf'))
    if len(figures) < 19:
        raise RuntimeError(f'Expected at least 19 scientific figure sets; found {len(figures)}.')
    for figure in figures:
        for suffix in ('.svg', '.png'):
            if not figure.with_suffix(suffix).is_file():
                raise RuntimeError(f'Missing figure representation: {figure.with_suffix(suffix)}')
    for path in (ROOT / 'results').glob('*.json'):
        json.loads(path.read_text())
    return dict(completed_pulses=len(jobs), figure_sets=len(figures), pulse_names=jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--content-only', action='store_true',
                        help='Check a deliberately regenerated tree without frozen byte hashes.')
    parser.add_argument('--public-package', action='store_true',
                        help='Explicitly select the public scientific package (the default).')
    parser.add_argument('--reviewer-package', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = content_checks()
    checked = 0
    if not args.content_only:
        manifest = ROOT / 'MANIFEST.sha256'
        if not manifest.is_file():
            raise RuntimeError('Manifest missing; use --content-only for a deliberately regenerated tree.')
        for line in manifest.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Checksum mismatch: ' + relative)
            checked += 1
    report.update(checksum_files=checked,
                  mode='content-only' if args.content_only else 'frozen-release',
                  package='public-scientific', status='passed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'Release check failed: {exc}', file=sys.stderr)
        sys.exit(1)
