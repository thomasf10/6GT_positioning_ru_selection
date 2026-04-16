import io
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplot2tikz


class _CompatUnpickler(pickle.Unpickler):
    """Redirect numpy._core.* -> numpy.core.* for pickles saved with NumPy >= 2.0."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def _safe_pickle_load(path):
    with open(path, 'rb') as f:
        data = f.read()
    return _CompatUnpickler(io.BytesIO(data)).load()


def evaluate_nmse_cdf(
    ue_ids,
    global_ru_id_labels,
    global_ru_id_predictions,
    results_csv_path,
    raw_data_path,
    num_rus_per_stripe=20,
    ue_beam_angle=0,
    ru_beam_angle=0,
    save_path=None,
    title='NMSE CDF: Optimal vs NN RU Selection',
):
    """
    Compare NMSE CDFs between optimal RU selection and NN-predicted RU selection.

    Args:
        ue_ids:                   list of UE IDs (actual IDs, not dataset indices)
        global_ru_id_labels:      list of ground-truth global RU IDs (stripe_id * num_rus_per_stripe + ru_id)
        global_ru_id_predictions: list of NN-predicted global RU IDs
        results_csv_path:         path to results.csv (contains optimal NMSE per UE)
        raw_data_path:            path to flickering_raw_data/ folder (per-UE pickle files)
        num_rus_per_stripe:       number of RUs per stripe (default 20)
        ue_beam_angle:            UE beam angle in degrees to filter on (default 0)
        ru_beam_angle:            RU beam angle in degrees to filter on (default 0)
        save_path:                if given, saves the plot to this path
        title:                    plot title

    Returns:
        (optimal_nmse_list, nn_nmse_list) — the two lists of NMSE values used for the CDFs
    """
    assert len(ue_ids) == len(global_ru_id_labels) == len(global_ru_id_predictions), \
        "ue_ids, labels, and predictions must have the same length"

    # Load results.csv; normalise column names to lowercase for robustness
    results_df = pd.read_csv(results_csv_path)
    results_df.columns = [c.lower() for c in results_df.columns]

    optimal_nmse_list = []
    optimal_fixed_beam_nmse_list = []
    best_ru_fixed_beam_nmse_list = []
    nn_nmse_list = []
    closest_fixed_beam_nmse_list = []
    closest_opt_beam_nmse_list = []
    skipped = 0

    for ue_id, label_global_ru, pred_global_ru in zip(ue_ids, global_ru_id_labels, global_ru_id_predictions):
        ue_id = int(ue_id)
        label_global_ru = int(label_global_ru)
        pred_global_ru = int(pred_global_ru)

        # --- Optimal NMSE from results.csv ---
        # Each ue_id in results.csv has exactly one row with the optimal RU and optimal beams
        opt_rows = results_df[(results_df['ue_id'] == ue_id)]

        if opt_rows.empty:
            skipped += 1
            continue
        opt_nmse = float(opt_rows['nmse'].iloc[0])

        # --- Closest RU IDs from results.csv ---
        closest_stripe = int(opt_rows['cstripe_id'].iloc[0])
        closest_ru = int(opt_rows['cru_id'].iloc[0])

        # --- NN NMSE from per-UE pickle ---
        pkl_path = os.path.join(raw_data_path, f'flickering_data_{ue_id}.pkl')
        if not os.path.exists(pkl_path):
            skipped += 1
            continue

        pkl_data = _safe_pickle_load(pkl_path)

        if isinstance(pkl_data, pd.DataFrame):
            pkl_df = pkl_data.copy()
        else:
            pkl_df = pd.DataFrame(pkl_data)
        pkl_df.columns = [c.lower() for c in pkl_df.columns]

        pred_stripe = pred_global_ru // num_rus_per_stripe
        pred_ru = pred_global_ru % num_rus_per_stripe

        nn_rows = pkl_df[
            (pkl_df['stripe_id'] == pred_stripe) &
            (pkl_df['ru_id'] == pred_ru) &
            (pkl_df['ue_beam_id'] == ue_beam_angle) &
            (pkl_df['ru_beam_id'] == ru_beam_angle)
        ]

        # --- Optimal RU with fixed beams (to isolate beam-selection gap) ---
        label_stripe = label_global_ru // num_rus_per_stripe
        label_ru = label_global_ru % num_rus_per_stripe

        opt_fixed_rows = pkl_df[
            (pkl_df['stripe_id'] == label_stripe) &
            (pkl_df['ru_id'] == label_ru) &
            (pkl_df['ue_beam_id'] == ue_beam_angle) &
            (pkl_df['ru_beam_id'] == ru_beam_angle)
        ]

        # --- Closest RU with fixed beams ---
        closest_fixed_rows = pkl_df[
            (pkl_df['stripe_id'] == closest_stripe) &
            (pkl_df['ru_id'] == closest_ru) &
            (pkl_df['ue_beam_id'] == ue_beam_angle) &
            (pkl_df['ru_beam_id'] == ru_beam_angle)
        ]

        # --- Closest RU with optimal beams (best NMSE across all beam combos) ---
        closest_all_beams = pkl_df[
            (pkl_df['stripe_id'] == closest_stripe) &
            (pkl_df['ru_id'] == closest_ru)
        ]

        # --- Best RU for fixed beams (lowest NMSE across all RUs at fixed beam angles) ---
        fixed_beam_all_rus = pkl_df[
            (pkl_df['ue_beam_id'] == ue_beam_angle) &
            (pkl_df['ru_beam_id'] == ru_beam_angle)
        ]

        if nn_rows.empty or opt_fixed_rows.empty or closest_fixed_rows.empty or closest_all_beams.empty or fixed_beam_all_rus.empty:
            skipped += 1
            continue
        nn_nmse = float(nn_rows['nmse'].iloc[0])
        opt_fixed_nmse = float(opt_fixed_rows['nmse'].iloc[0])
        closest_fixed_nmse = float(closest_fixed_rows['nmse'].iloc[0])
        closest_opt_nmse = float(closest_all_beams['nmse'].min())
        best_ru_fixed_nmse = float(fixed_beam_all_rus['nmse'].min())

        optimal_nmse_list.append(opt_nmse)
        optimal_fixed_beam_nmse_list.append(opt_fixed_nmse)
        best_ru_fixed_beam_nmse_list.append(best_ru_fixed_nmse)
        nn_nmse_list.append(nn_nmse)
        closest_fixed_beam_nmse_list.append(closest_fixed_nmse)
        closest_opt_beam_nmse_list.append(closest_opt_nmse)

    print(f"Evaluated {len(optimal_nmse_list)} UEs, skipped {skipped} (missing data)")

    if not optimal_nmse_list:
        print("No valid samples to plot.")
        return [], [], [], [], [], []

    def compute_cdf(values):
        sorted_v = np.sort(values)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        return sorted_v, cdf

    opt_sorted, opt_cdf = compute_cdf(optimal_nmse_list)
    opt_fixed_sorted, opt_fixed_cdf = compute_cdf(optimal_fixed_beam_nmse_list)
    best_ru_fixed_sorted, best_ru_fixed_cdf = compute_cdf(best_ru_fixed_beam_nmse_list)
    closest_opt_sorted, closest_opt_cdf = compute_cdf(closest_opt_beam_nmse_list)
    closest_fixed_sorted, closest_fixed_cdf = compute_cdf(closest_fixed_beam_nmse_list)
    nn_sorted, nn_cdf = compute_cdf(nn_nmse_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(opt_sorted, opt_cdf, label='Optimal RU, optimal beams', linewidth=2)
    ax.plot(opt_fixed_sorted, opt_fixed_cdf, label=f'Optimal RU, fixed beams ({ue_beam_angle}°/{ru_beam_angle}°)', linewidth=2, linestyle='-.')
    ax.plot(best_ru_fixed_sorted, best_ru_fixed_cdf, label=f'Best RU for fixed beams ({ue_beam_angle}°/{ru_beam_angle}°)', linewidth=2, linestyle='-.')
    ax.plot(closest_opt_sorted, closest_opt_cdf, label='Closest RU, optimal beams', linewidth=2, linestyle='-.')
    ax.plot(closest_fixed_sorted, closest_fixed_cdf, label=f'Closest RU, fixed beams ({ue_beam_angle}°/{ru_beam_angle}°)', linewidth=2, linestyle=':')
    ax.plot(nn_sorted, nn_cdf, label='NN selected RU', linewidth=2, linestyle='--')
    ax.set_xlabel('NMSE (dB)')
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        tikz_path = os.path.splitext(save_path)[0] + '.tex'
        matplot2tikz.save(tikz_path)
        print(f"TikZ saved to {tikz_path}")
    plt.show()

    return optimal_nmse_list, optimal_fixed_beam_nmse_list, best_ru_fixed_beam_nmse_list, closest_opt_beam_nmse_list, closest_fixed_beam_nmse_list, nn_nmse_list


# ---------------------------------------------------------------------------
# Default paths relative to this script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_ROOT = os.path.join(_SCRIPT_DIR, '..', 'dataset', 'office_space_inline', 'ru_selection_labels')

DEFAULT_RESULTS_CSV = os.path.join(_DATASET_ROOT, 'results.csv')
DEFAULT_RAW_DATA = os.path.join(_DATASET_ROOT, 'flickering_raw_data')


if __name__ == '__main__':
    # Quick smoke-test: load a few UEs from results.csv and use their labels as
    # both labels AND predictions (perfect selection) — the two CDFs should overlap.
    df = pd.read_csv(DEFAULT_RESULTS_CSV)
    df.columns = [c.lower() for c in df.columns]

    demo = df[(df['ue_beam_id'] == 0) & (df['ru_beam_id'] == 0)].head(50)
    sample_ue_ids   = demo['ue_id'].tolist()
    sample_labels   = (demo['stripe_id'] * 20 + demo['ru_id']).tolist()
    sample_preds    = sample_labels  # perfect predictions for smoke-test

    evaluate_nmse_cdf(
        sample_ue_ids,
        sample_labels,
        sample_preds,
        DEFAULT_RESULTS_CSV,
        DEFAULT_RAW_DATA,
        save_path=os.path.join(_SCRIPT_DIR, 'pdp32_results', 'nmse_cdf_test.png'),
    )
