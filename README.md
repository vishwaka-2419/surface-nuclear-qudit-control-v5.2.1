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

A GitHub URL and Zenodo DOI must be assigned through the corresponding services. No identifier is invented in this prepared checkout. The licensing decision is also pending; see `LICENSE_NOTICE.md`. These release details must be finalized before claiming that the materials are publicly available under a reuse licence.

## What the paper shows


1. **Optimized multitone gates and composability.** For a direct nuclear drive and a static electron, fourteen-tone shaped radio-frequency pulses implement an eight-level collective π/2 rotation in 12 µs with fidelity > 99.99 % in the full 16-level model at 1.03, 1.35, 1.6 and 1.714 T as isolated gates at their design start time. A seven-tone pulse for a prepared electron is also validated at 1.6 T, reaching 99.9993 %. A pulse replayed at another time with continuously running carriers keeps its resonant couplings but rephases the off-resonant ones: near the comb crossing at 1.35 T the nominal pulse falls to 45 % and two consecutive pulses to 50 %. Pulses trained on start times drawn from 0–100 ms sample a broader range of relative phases and are validated at 48 held-out start times, on a 400-point scan and in two- and three-pulse sequences; the numbers are in `docs/FINAL_RESULTS.md`. These are finite sampled tests, not a guarantee for every carrier phase or sequence. Pulses trained on a 10-µs window pass the sampled tests inside that window but can degrade at later starts (SI Supplementary Note 9).
2. **Driving through the electron.** For an electron-mediated drive, one plausible hypothesis for the junction, the drive shifts the whole nuclear transition comb by Δf ≈ A_z⟨Ω_e²⟩/4ν_e² (9–23 kHz for the pulses considered), comparable to the nuclear Rabi frequency. A pulse designed without the shift reaches F ≈ 0.1–0.2 in the full model although the electron returns to its initial state. Calibrating the shift into the carrier frequencies (frame defined by the carriers) and designing with the second-order model restores F = 99.87–99.997 % (seven-tone pulses ≥ 99.96 %). The leading formula assumes an effective spin-1/2 electron, ν_e ≫ A_z and Ω_e ≪ ν_e.
3. **Calibration tolerance.** With a residual-error domain of ±0.1 mT, ±2 kHz (A_z, κ) and ±3 % gain, a 48-µs corner-trained direct-drive pulse keeps 99.3 % at its worst corner and 99.45 % at the worst of 128 held-out points. The frozen shift-tracked electron-mediated pulses are more sensitive (98.4 % / 95.8 % at ±3 % gain for 7 / 14 tones; 92.6 % at the worst corner for 7 tones), because the comb shift scales with drive power.
4. **Electron switching.** A classical telegraph model and a sufficient (not necessary) bound convert gate duration, electron correlation time and error budget into a maximum stationary excited fraction: p_e ≤ 8.5×10⁻⁵ for T = 12 µs, τ_c = 0.1 µs and a 0.1 % coherent-error allowance (≈ 67 mK thermal equivalent at 1.6 T); with the 48-µs robust pulse's actual 0.7 % coherent error, p_e ≤ 7.0×10⁻⁶. Motional narrowing would need femtosecond correlation times. Nuclear dephasing requirements (T_φ ≳ 11 ms magnetic, T_Q ≳ 1.2 ms quadrupolar) are given separately. Every result in the paper is labeled with its validation level (L1 design mesh, L2 real waveform, L3 second-order model, L4 full 16-level model).


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

## Fresh optimization and numerical studies

See `docs/REPRODUCING.md` for the ordered study recipe, including the 1.03 T continuation and the 24-start-time refined 1.6 T run. Work in a separate copy. Fresh optimization takes hours, and floating-point/library differences can change the trajectory of a finite-budget optimizer. The archived coefficient arrays are the exact controls used for the reported results.

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
| `figures/` | PDF, PNG and SVG scientific figures |
| `manuscript/generated/` | Generated numerical macros used in the article |
| `supplement/generated/` | Generated numerical tables used in the supplement |
| `tests/` | Algebra, signs, frames, gradients, fidelity, channel and propagation checks |
| `configs/` | Experimental calibration interface |
| `docs/` | Scientific limits, numerical findings, AI disclosure, calibration and reproduction instructions |
| `provenance/` | Verification records, input hashes, figure manifest and clearly labelled historical records |

## Citation and reuse

`CITATION.cff` supplies the author, title and numerical version. Cite the specific archived version once a real DOI has been registered. Do not use an unassigned DOI or describe an unpublished draft as a public archive. No public reuse licence has yet been assigned; `LICENSE_NOTICE.md` records the current rights status.

AI assistance is disclosed in the manuscript Methods and `docs/AI_ASSISTANCE.md`. The author is responsible for the scientific claims, code and interpretation.
