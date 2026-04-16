"""
Compute RU selection labels for fixed 0-degree beams.

For each UE, finds the RU with minimum NMSE when both UE and RU beams
are fixed at 0 degrees, using the per-UE pickle files.

Output: a CSV with columns matching results.csv format so it can be
used directly as `labels_path` in CsiDataset.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from evaluation import load_ue_pickle


def compute_fixed_beam_labels(
    results_csv_path,
    raw_data_path,
    output_csv_path,
    ue_beam_angle=0,
    ru_beam_angle=0,
):
    """
    For every UE in results.csv, find the best RU at fixed beam angles.

    Parameters
    ----------
    results_csv_path : str — path to the original results.csv
    raw_data_path : str — directory with flickering_data_*.pkl files
    output_csv_path : str — where to write the new labels CSV
    ue_beam_angle : int — fixed UE beam angle (degrees)
    ru_beam_angle : int — fixed RU beam angle (degrees)
    """
    results_df = pd.read_csv(results_csv_path)
    results_df.columns = [c.lower() for c in results_df.columns]

    rows = []
    skipped = 0

    for _, orig_row in tqdm(results_df.iterrows(), total=len(results_df), desc='Computing labels'):
        ue_id = int(orig_row['ue_id'])

        pkl_df = load_ue_pickle(raw_data_path, ue_id)
        if pkl_df is None:
            skipped += 1
            continue

        fixed = pkl_df[
            (pkl_df['ue_beam_id'] == ue_beam_angle) &
            (pkl_df['ru_beam_id'] == ru_beam_angle)
        ]

        if fixed.empty:
            skipped += 1
            continue

        best_idx = fixed['nmse'].idxmin()
        best = fixed.loc[best_idx]

        row = {
            'ue_id': ue_id,
            'stripe_id': int(best['stripe_id']),
            'ru_id': int(best['ru_id']),
            'ue_beam_id': ue_beam_angle,
            'ru_beam_id': ru_beam_angle,
            'nmse': float(best['nmse']),
        }
        # Carry over position columns from original results.csv
        for col in ['uex', 'uey', 'uez', 'rux', 'ruy', 'ruz']:
            if col in orig_row:
                row[col] = orig_row[col]

        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f'Wrote {len(out_df)} labels to {output_csv_path}  (skipped {skipped})')


if __name__ == '__main__':
    dataset_folder = 'office_space_reduced_inline'
    base = Path(__file__).parent.parent / 'dataset' / dataset_folder / 'ru_selection_labels'

    compute_fixed_beam_labels(
        results_csv_path=str(base / 'results.csv'),
        raw_data_path=str(base / 'flickering_raw_data'),
        output_csv_path=str(base / 'results_fixed_beams_0deg.csv'),
    )
