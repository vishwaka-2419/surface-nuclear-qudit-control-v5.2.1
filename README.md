# Requirements for coherent control of a single-atom nuclear-spin qudit

**Version 5.2.1 — 20 October 2026**

Research code, numerical data, optimized pulse coefficients, figures and reproducibility instructions for the computational study by **Aishwarya Vishwakarma**, Department of Quantum Matter Physics, University of Geneva.

The study models coherent control of the `I = 7/2` nucleus of a single 49Ti atom on MgO/Ag(100), using the full 16-level electron–nuclear Hamiltonian with published parameters. The results are model calculations, not measurements of an operating device. The scientific assumptions and limitations are summarized in `docs/CLAIMS_AND_LIMITATIONS.md`.

This repository contains the code and data needed to reproduce the numerical results and figures. Manuscript prose, paper PDFs and private submission materials are not included. The numerical source files, result arrays, pulse coefficients, figure data and figure files are preserved from the verified v5.2.1 author package. Repository documentation and packaging scripts have been adapted for public use. The folder can be uploaded to GitHub after extraction; add your selected reuse licence and real archive DOI when you create the public release.

## Start here

1. Install the Python dependencies below.
2. Run `python scripts/check_release.py` to verify the frozen archive before regenerating files.
3. Run `python -m pytest -q` for the assertion tests.
4. Regenerate the figures from the supplied results, or independently propagate a saved pulse, using the commands below.

## Installation

Use Python 3.11 or later. The original numerical environments and the separate Python 3.12 verification environment are recorded in `provenance/historical/environment_pre_v521.json` and `provenance/audit_environment_v521.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e .
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

On Windows, activate the virtual environment with the equivalent command for the selected shell. `requirements-tested.txt` specifies dependency ranges; the provenance records identify the exact versions used in the recorded runs. `pyproject.toml` declares the numerical dependencies and optional test dependencies.

## Verify the archive and tests

```bash
python scripts/check_release.py
python -m pytest -q
```

The release checker validates required scientific files, completed pulse metadata, finite coefficient arrays, figure representations, JSON readability and the SHA-256 manifest. The frozen v5.2.1 numerical set contains 39 pulse files and 19 figure sets.

The recorded final audit passed 63 parameter-expanded assertion cases using `scripts/audit_assertions_fallback.py`, because pytest was unavailable in that audit environment. That record is explicitly a direct assertion run, not a pytest invocation. The ordinary public workflow above uses pytest. The fallback script is retained for traceability; running it rewrites `provenance/final_assertion_results.json`, so use it only in a working copy.

After intentionally regenerating files, their byte hashes may change. Use `python scripts/check_release.py --content-only` to check the regenerated tree; preserve the original archive and its manifest for comparison.

## Regenerate figures, tables and numerical macros

The supplied `results/` files are sufficient for this route; no pulse optimization is required.

```bash
python scripts/comb_spectrum.py
python scripts/build_figures_tables.py
```

The figure script writes `figures/`, `figure_data/`, `supplement/generated/` and `manuscript/generated/numbers.tex`. `provenance/figure_manifest.csv` maps each figure to its source result files. The figures are deterministic renderings of numerical data, although PDF metadata or library-version differences can change file bytes.

## Independently verify a saved pulse

```bash
python scripts/verify_pulse.py nominal_N_B1.6_dual
python scripts/verify_pulse.py composable_N_B1.6_dual --start-time-us 769.615835
python scripts/verify_pulse.py tracked_electron_B1.6_single --full-step-us 0.000025
```

These commands propagate archived waveforms without re-optimizing them. New reports go to `verification/`, preserving `results/`. For electron-mediated pulses, compare the second-order reduced model and full 16-level model; disagreement with the linearly projected model is part of the physical result. Fidelity differences and step convergence are the appropriate comparisons, not byte-identical output JSON.

## Export coefficient waveforms

```bash
python scripts/export_waveform.py tracked_electron_B1.6_single --output exports/nuclear_gate.csv
```

The output CSV contains time in microseconds and the real Hamiltonian coefficient in MHz. The companion files record complex envelopes, carrier frequencies, operator normalization and frame convention. These are not calibrated instrument voltages. `docs/EXPERIMENTAL_CALIBRATION.md` and `configs/experimental_interface.json` describe the quantities required for a physical implementation.

## Units and conventions

Energy divided by Planck's constant is in MHz; time is in microseconds; magnetic field is in tesla. Evolution is `exp(-2πi H t)`. The positive nuclear Zeeman coefficient is `gamma = -g_N μ_N/h`. The reference parameter set is `extended.nominal_parameters()`.

Carrier offsets define the conditional frame of the shift-tracked targets. A pulse start time specifies when the envelope begins while carriers continue to run from laboratory time zero. The source-motivated parameter set is a composite from different atoms; source uncertainty and residual local calibration error are distinct. See `docs/CLAIMS_AND_LIMITATIONS.md` for the evidential boundaries.

## Directory map

| Path | Content |
|---|---|
| `src/qudit_control/` | Hamiltonian, spline controls, second-order model, independent propagators, noise channels and requirements |
| `scripts/` | Study drivers, figure generation, pulse verification, waveform export and public release checks |
| `results/` | Numerical results, scan records, optimizer metadata and `pulses/*.npz` coefficient arrays |
| `figure_data/` | Data associated with the plotted figures |
| `manuscript/generated/` | Generated numerical macros used in the article |
| `supplement/generated/` | Generated numerical tables used in the supplement |
| `tests/` | Algebra, signs, frames, gradients, fidelity, channel and propagation checks |
| `configs/` | Experimental calibration interface |
| `provenance/` | Verification records, input hashes, figure manifest and clearly labelled historical records |
