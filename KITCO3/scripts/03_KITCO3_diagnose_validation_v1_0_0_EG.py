#!/usr/bin/env python3
"""Read-only diagnostic for the Script03 residual-IBI validation failure.

Version 1.0.0. Reads existing generated results, checks reference hashes, and
compares SciPy's local median with an explicit 11-point edge-padded median.
Does not change data, QC decisions, files, environments or validation rules.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import median_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path,
                        default=Path('/Volumes/One Touch/KITCO3_drive'))
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    ref = root / 'scripts/KITCO3_pipeline_02_03_04_v1.0.0/support/ecg_review'
    out = root / 'outputs/03_ecg_QC_v2_1_0/historical_qc/results'
    report = {'diagnostic_version': '1.0.0', 'read_only': True,
              'versions': {'python': platform.python_version(),
                           'numpy': np.__version__, 'pandas': pd.__version__,
                           'scipy': scipy.__version__}}
    seal = json.loads((ref / 'results/OUTPUT_SHA256.json').read_text())
    report['reference_file_mismatches'] = [
        name for name, expected in seal.items()
        if not (out / name).is_file()
        or hashlib.sha256((out / name).read_bytes()).hexdigest() != expected]
    ibi = pd.read_csv(out / 'reviewed_nonbridging_ibi.csv.gz')
    records = pd.read_csv(ref / 'inputs/recordings_142.csv')
    counts = {'invalid_trusted_IBIs': 0, 'median_disagreements': 0,
              'scipy_residual_flags': 0, 'explicit_residual_flags': 0}
    examples = []
    for r in records.itertuples(index=False):
        duration = r.cooking_stop_timestamp - r.cooking_start_timestamp
        good = ibi.loc[
            ibi.recording_uid.eq(r.recording_uid) & ibi.trusted
            & ibi.midpoint_cooking_s.ge(0) & ibi.midpoint_cooking_s.lt(duration)]
        for seg, s in good.groupby('ibi_segment_id'):
            v = s.ibi_s.to_numpy(dtype=float)
            invalid = ~np.isfinite(v) | (v <= 0)
            counts['invalid_trusted_IBIs'] += int(invalid.sum())
            if invalid.any():
                continue
            local = median_filter(v, size=11, mode='nearest')
            padded = np.pad(v, (5, 5), mode='edge')
            explicit = np.median(
                np.lib.stride_tricks.sliding_window_view(padded, 11), axis=1)
            disagrees = local != explicit
            counts['median_disagreements'] += int(disagrees.sum())
            if disagrees.any() and len(examples) < 5:
                j = int(np.flatnonzero(disagrees)[0])
                examples.append({'recording_uid': r.recording_uid,
                                 'segment_id': int(seg), 'segment_IBIs': len(v),
                                 'ibi_s': float(v[j]), 'scipy_median_s': float(local[j]),
                                 'explicit_median_s': float(explicit[j])})
            for label, median in [('scipy', local), ('explicit', explicit)]:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = v / median
                counts[label + '_residual_flags'] += int(
                    ((ratio < .55) | (ratio > 1.7) | (v < .3) | (v > 2.0)).sum())
    report['counts'] = counts
    report['median_disagreement_examples'] = examples
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
