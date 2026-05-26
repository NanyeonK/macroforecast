# tools/research -- Research-phase scripts

This directory contains scripts written during active research that produced
data structures now committed to the main codebase. They are preserved as an
audit trail and are NOT part of the build system, CI pipeline, or installed
package.

## Contents

### `build_sd_tcode_validation.py`

Generates diagnostic validation artifacts for the inferred FRED-SD t-code
candidates in `macroforecast/layers/l1_data/sd_inferred_tcodes.py` and
`sd_analog_candidates.py`. Produces correlation statistics between inferred
and official t-code transform series.

This script was used to validate the `SD_INFERRED_TCODE_MAP` and
`SD_ANALOG_CANDIDATES` data structures before they were committed. It is
retained as documentation of that validation process.

To run (requires FRED-SD data locally):
```
python tools/research/build_sd_tcode_validation.py --help
```

## Policy

Do not import from this directory in tests or production code.
Do not add new research scripts here without a corresponding README entry.
When a research script's output is committed to the codebase, annotate this
file with the commit hash where that happened.
