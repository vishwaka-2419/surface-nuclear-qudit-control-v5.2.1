# Provenance of the public scientific package

These records document distinct preparation and validation stages. Historical files retain their original versions and counts; they do not describe the current release totals.

## Current v5.2.1 scientific audit

- `numerical_audit_v521.json`: scientific fields from the 20 October 2026 final author audit. The earlier 61-case test run and the final 63-case run are separately recorded. Review-mode and pending-administration fields have been omitted in this public subset; numerical results are unchanged.
- `audit_environment_v521.json`: the Python 3.12 numerical environment used for selected independent checks and the final assertion harness. The complete optimization campaign was not rerun.
- `final_assertion_results.json`: 63 parameter-expanded assertion cases, all passing, executed directly because pytest was unavailable in that audit environment.
- `full_sequence_refresh.json` and `full_sequence_patch_validation.json`: the complete-propagator sequence correction and refreshed results.
- `continuous_two_pulse_full16.json`: a continuous full-model two-pulse propagation comparison.
- `figure_manifest.csv`: figure-to-result mapping.
- `input_hashes_v521.json`: scientific source/input hashes from that audit for files retained in this code/data package.

## Historical records

- `historical/environment_pre_v521.json`: environments reported for the original v4 numerical work and subsequent v5 tracked designs/figures.
- `historical/execution_manifest_v4.json`: historical v4 scientific execution summary, with its original 23-pulse/56-test/18-figure statistics. Administrative PDF counts are omitted. These are not the final v5.2.1 totals.
- `historical/assertion_results_pre_v521.json`: the earlier 61-case assertion check, before the two added full-propagator interface checks.
- `historical/fresh_pulse_check_supplied_v52.json`: a selected pulse check supplied with the earlier v5.2 material.

The public package preparation preserves the numerical source, results, figure data and figure bytes from the verified v5.2.1 source package. A new top-level `MANIFEST.sha256` identifies the final public tree. New public-release checks, identifiers and metadata may be recorded separately without rewriting historical numerical evidence.
