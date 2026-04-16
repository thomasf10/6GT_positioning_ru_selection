"""
Evaluation utilities for RU selection models.

Contains helpers for loading per-UE data, computing NMSE metrics,
and plotting CDF curves. Import individual functions as needed:

    from evaluation import evaluate_nmse_cdf, compute_cdf, load_ue_pickle
"""
import io
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplot2tikz


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_ue_pickle(raw_data_path, ue_id):
    """Load and return a normalised DataFrame from a per-UE pickle file.

    Returns None if the file does not exist.
    """
    pkl_path = os.path.join(raw_data_path, f'flickering_data_{ue_id}.pkl')
    if not os.path.exists(pkl_path):
        return None

    pkl_data = _safe_pickle_load(pkl_path)
    if isinstance(pkl_data, pd.DataFrame):
        pkl_df = pkl_data.copy()
    else:
        pkl_df = pd.DataFrame(pkl_data)
    pkl_df.columns = [c.lower() for c in pkl_df.columns]
    return pkl_df


def compute_cdf(values, max_points=200):
    """Return (sorted_values, cdf) arrays, downsampled to *max_points*.

    If the input has more than *max_points* entries the CDF is linearly
    interpolated onto an evenly spaced grid to keep TikZ files compact.
    """
    sorted_v = np.sort(values)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    if max_points is not None and len(sorted_v) > max_points:
        v_grid = np.linspace(sorted_v[0], sorted_v[-1], max_points)
        cdf = np.interp(v_grid, sorted_v, cdf)
        sorted_v = v_grid
    return sorted_v, cdf


def save_figure(fig, save_path):
    """Save a matplotlib figure as PNG and TikZ (.tex)."""
    if save_path is None:
        return
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    tikz_path = os.path.splitext(save_path)[0] + '.tex'
    code = matplot2tikz.get_tikz_code()
    with open(tikz_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"TikZ saved to {tikz_path}")


def get_ru_positions(config_path):
    """Compute all RU (x, y) positions from a dataset config YAML."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sc = cfg['stripe_config']
    start = sc['stripe_start_pos']   # [x, y_start, z]
    end = sc['stripe_end_pos']       # [x, y_end,   z]
    n_stripes = sc['N_stripes']
    n_rus = sc['N_RUs']
    spacing_ru = sc['space_between_RUs']
    spacing_stripe = sc['space_between_stripes']
    direction = sc.get('stripe_direction', 'y')

    ru_x, ru_y = [], []
    for s in range(n_stripes):
        for r in range(n_rus):
            if direction == 'y':
                x = start[0] + s * spacing_stripe
                y = start[1] + r * spacing_ru
            else:
                x = start[0] + r * spacing_ru
                y = start[1] + s * spacing_stripe
            ru_x.append(x)
            ru_y.append(y)
    return ru_x, ru_y


# ---------------------------------------------------------------------------
# Spatial NMSE delta plot
# ---------------------------------------------------------------------------

def plot_spatial_nmse_delta(
    ue_x,
    ue_y,
    nmse_a,
    nmse_b,
    ru_x,
    ru_y,
    label_a='A',
    label_b='B',
    save_path=None,
    title=None,
):
    """
    Plot the per-UE NMSE difference (nmse_b - nmse_a) on a 2D scatter map
    with RU locations overlaid.

    Parameters
    ----------
    ue_x, ue_y : array-like  — UE positions (metres).
    nmse_a, nmse_b : array-like  — per-UE NMSE values (dB).
    ru_x, ru_y : array-like  — RU positions for overlay.
    label_a, label_b : str  — legend labels for the two strategies.
    save_path : str or None  — where to save the figure.
    title : str or None  — plot title override.
    """
    ue_x = np.asarray(ue_x)
    ue_y = np.asarray(ue_y)
    delta = np.asarray(nmse_b) - np.asarray(nmse_a)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ue_x, ue_y, c=delta, cmap='RdBu_r', s=12, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f'NMSE delta (dB)\n({label_b} − {label_a})')

    ax.scatter(ru_x, ru_y, marker='^', c='black', s=80, zorder=5, label='RU locations')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title or f'Spatial NMSE delta: {label_b} − {label_a}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, save_path)
    plt.show()


def plot_ue_ru_locations(
    ue_ids,
    results_csv_path,
    ru_x,
    ru_y,
    save_path=None,
    title='UE and RU locations',
):
    """
    Scatter-plot UE positions with RU locations overlaid.

    UE positions are looked up from results.csv by ue_id.
    """
    results_df = pd.read_csv(results_csv_path)
    results_df.columns = [c.lower() for c in results_df.columns]

    ue_set = set(int(u) for u in ue_ids)
    ue_rows = results_df[results_df['ue_id'].isin(ue_set)].drop_duplicates(subset='ue_id')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ue_rows['uex'], ue_rows['uey'], s=8, alpha=0.5, label=f'UE ({len(ue_rows)})')
    ax.scatter(ru_x, ru_y, marker='^', c='black', s=80, zorder=5, label='RU locations')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, save_path)
    plt.show()


# ---------------------------------------------------------------------------
# NMSE CDF evaluation
# ---------------------------------------------------------------------------

def evaluate_nmse_cdf(
    ue_ids,
    global_ru_id_labels,
    global_ru_id_predictions,
    results_csv_path,
    raw_data_path,
    num_rus_per_stripe=20,
    ue_beam_angle=0,
    ru_beam_angle=0,
    nn_ue_beam_angles=None,
    nn_ru_beam_angles=None,
    save_path=None,
    title='NMSE CDF: Optimal vs NN RU Selection',
):
    """
    Compare NMSE CDFs between various RU selection strategies and NN predictions.

    Plots six curves:
      1. Optimal RU, optimal beams  (upper bound from results.csv)
      2. Optimal RU, fixed beams    (optimal RU but beams locked to ue/ru_beam_angle)
      3. Best RU for fixed beams    (best RU when beams are fixed, computed from pkl)
      4. Closest RU, optimal beams  (geographically closest RU, best beam combo)
      5. Closest RU, fixed beams    (geographically closest RU, beams locked)
      6. NN selected RU + beams     (model prediction; per-UE beams if provided,
                                     otherwise beams locked to ue/ru_beam_angle)

    Parameters
    ----------
    nn_ue_beam_angles, nn_ru_beam_angles : list[int] or None
        Per-UE predicted beam **angles** (e.g. -30, 0, 10 …) for the NN curve.
        When *None* the NN curve uses the fixed ``ue_beam_angle`` / ``ru_beam_angle``.

    Returns:
        dict with keys: 'optimal', 'optimal_fixed', 'best_ru_fixed',
        'closest_optimal', 'closest_fixed', 'nn' — each a list of NMSE values.
    """
    assert len(ue_ids) == len(global_ru_id_labels) == len(global_ru_id_predictions), \
        "ue_ids, labels, and predictions must have the same length"
    nn_has_beams = nn_ue_beam_angles is not None and nn_ru_beam_angles is not None
    if nn_has_beams:
        assert len(nn_ue_beam_angles) == len(ue_ids), \
            "nn_ue_beam_angles must have same length as ue_ids"
        assert len(nn_ru_beam_angles) == len(ue_ids), \
            "nn_ru_beam_angles must have same length as ue_ids"

    results_df = pd.read_csv(results_csv_path)
    results_df.columns = [c.lower() for c in results_df.columns]

    nmse = {
        'optimal': [],
        'optimal_fixed': [],
        'best_ru_fixed': [],
        'closest_optimal': [],
        'closest_fixed': [],
        'nn': [],
        'ue_x': [],
        'ue_y': [],
    }
    skipped = 0

    iter_items = zip(ue_ids, global_ru_id_labels, global_ru_id_predictions,
                     nn_ue_beam_angles if nn_has_beams else [None] * len(ue_ids),
                     nn_ru_beam_angles if nn_has_beams else [None] * len(ue_ids))

    for ue_id, label_global_ru, pred_global_ru, nn_ue_ba, nn_ru_ba in iter_items:
        ue_id = int(ue_id)
        label_global_ru = int(label_global_ru)
        pred_global_ru = int(pred_global_ru)

        # --- Optimal NMSE from results.csv ---
        opt_rows = results_df[results_df['ue_id'] == ue_id]
        if opt_rows.empty:
            skipped += 1
            continue
        opt_nmse = float(opt_rows['nmse'].iloc[0])
        closest_stripe = int(opt_rows['cstripe_id'].iloc[0])
        closest_ru = int(opt_rows['cru_id'].iloc[0])

        # --- Per-UE pickle ---
        pkl_df = load_ue_pickle(raw_data_path, ue_id)
        if pkl_df is None:
            skipped += 1
            continue

        pred_stripe = pred_global_ru // num_rus_per_stripe
        pred_ru = pred_global_ru % num_rus_per_stripe
        label_stripe = label_global_ru // num_rus_per_stripe
        label_ru = label_global_ru % num_rus_per_stripe

        fixed_beam_mask = (
            (pkl_df['ue_beam_id'] == ue_beam_angle) &
            (pkl_df['ru_beam_id'] == ru_beam_angle)
        )

        # NN predicted RU (+ predicted beams if available, else fixed beams)
        if nn_has_beams:
            nn_beam_mask = (
                (pkl_df['ue_beam_id'] == int(nn_ue_ba)) &
                (pkl_df['ru_beam_id'] == int(nn_ru_ba))
            )
        else:
            nn_beam_mask = fixed_beam_mask

        nn_rows = pkl_df[
            (pkl_df['stripe_id'] == pred_stripe) &
            (pkl_df['ru_id'] == pred_ru) &
            nn_beam_mask
        ]

        # Optimal RU, fixed beams
        opt_fixed_rows = pkl_df[
            (pkl_df['stripe_id'] == label_stripe) &
            (pkl_df['ru_id'] == label_ru) &
            fixed_beam_mask
        ]

        # Closest RU, fixed beams
        closest_fixed_rows = pkl_df[
            (pkl_df['stripe_id'] == closest_stripe) &
            (pkl_df['ru_id'] == closest_ru) &
            fixed_beam_mask
        ]

        # Closest RU, optimal beams (min NMSE across all beam combos)
        closest_all_beams = pkl_df[
            (pkl_df['stripe_id'] == closest_stripe) &
            (pkl_df['ru_id'] == closest_ru)
        ]

        # Best RU for fixed beams (min NMSE across all RUs at fixed beams)
        fixed_beam_all_rus = pkl_df[fixed_beam_mask]

        if any(df.empty for df in [nn_rows, opt_fixed_rows, closest_fixed_rows, closest_all_beams, fixed_beam_all_rus]):
            skipped += 1
            continue

        nmse['optimal'].append(opt_nmse)
        nmse['optimal_fixed'].append(float(opt_fixed_rows['nmse'].iloc[0]))
        nmse['best_ru_fixed'].append(float(fixed_beam_all_rus['nmse'].min()))
        nmse['closest_optimal'].append(float(closest_all_beams['nmse'].min()))
        nmse['closest_fixed'].append(float(closest_fixed_rows['nmse'].iloc[0]))
        nmse['nn'].append(float(nn_rows['nmse'].iloc[0]))
        nmse['ue_x'].append(float(opt_rows['uex'].iloc[0]))
        nmse['ue_y'].append(float(opt_rows['uey'].iloc[0]))

    print(f"Evaluated {len(nmse['optimal'])} UEs, skipped {skipped} (missing data)")

    if not nmse['optimal']:
        print("No valid samples to plot.")
        return nmse

    # Build CDF plot
    curves = [
        ('optimal',         'Optimal RU, optimal beams',                              '-',  None),
        ('optimal_fixed',   f'Optimal RU, fixed beams ({ue_beam_angle}°/{ru_beam_angle}°)',  '-.', None),
        ('best_ru_fixed',   f'Best RU for fixed beams ({ue_beam_angle}°/{ru_beam_angle}°)',  '-.', None),
        ('closest_optimal', 'Closest RU, optimal beams',                              '-.', None),
        ('closest_fixed',   f'Closest RU, fixed beams ({ue_beam_angle}°/{ru_beam_angle}°)',  ':',  None),
        ('nn',              'NN selected RU + beams' if nn_has_beams else 'NN selected RU',  '--', None),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, linestyle, _color in curves:
        sorted_v, cdf = compute_cdf(nmse[key])
        ax.plot(sorted_v, cdf, label=label, linewidth=2, linestyle=linestyle)

    ax.set_xlabel('NMSE (dB)')
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, save_path)
    plt.show()

    return nmse


# ---------------------------------------------------------------------------
# Default paths (for standalone testing)
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
    sample_ue_ids = demo['ue_id'].tolist()
    sample_labels = (demo['stripe_id'] * 20 + demo['ru_id']).tolist()
    sample_preds = sample_labels  # perfect predictions for smoke-test

    evaluate_nmse_cdf(
        sample_ue_ids,
        sample_labels,
        sample_preds,
        DEFAULT_RESULTS_CSV,
        DEFAULT_RAW_DATA,
        save_path=os.path.join(_SCRIPT_DIR, 'pdp32_results', 'nmse_cdf_test.png'),
    )
