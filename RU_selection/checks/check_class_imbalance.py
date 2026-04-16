import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add workspace root to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.dataloaders import CsiDataset


def summarize_counts(name, counts, num_classes):
    total_samples = sum(counts.values())
    present_classes = sorted(counts.keys())
    missing_classes = [class_id for class_id in range(num_classes) if class_id not in counts]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Samples: {total_samples}")
    print(f"Classes present: {len(present_classes)}/{num_classes}")
    print(f"Classes missing: {len(missing_classes)}")

    if missing_classes:
        preview = missing_classes[:20]
        print(f"Missing class IDs (first 20): {preview}")

    if counts:
        count_values = np.array(list(counts.values()), dtype=np.int64)
        max_count = int(count_values.max())
        min_count = int(count_values.min())
        mean_count = float(count_values.mean())
        median_count = float(np.median(count_values))
        imbalance_ratio = float(max_count / max(min_count, 1))

        print(f"Max class count: {max_count}")
        print(f"Min non-zero class count: {min_count}")
        print(f"Mean class count: {mean_count:.2f}")
        print(f"Median class count: {median_count:.2f}")
        print(f"Imbalance ratio (max/min non-zero): {imbalance_ratio:.2f}")

        most_common = counts.most_common(10)
        least_common = sorted(counts.items(), key=lambda item: item[1])[:10]

        print("Most common classes (top 10):")
        for class_id, class_count in most_common:
            print(f"  global_ru_id={class_id:3d}: {class_count:4d}")

        print("Least common classes (bottom 10 non-zero):")
        for class_id, class_count in least_common:
            print(f"  global_ru_id={class_id:3d}: {class_count:4d}")


def counts_from_subset(subset):
    base_dataset = subset.dataset
    counts = Counter()
    for sample_index in subset.indices:
        user_id = base_dataset.valid_users[sample_index]
        label = base_dataset.labels_dict.get(user_id)
        if label is None:
            continue
        counts[label['global_ru_id']] += 1
    return counts


def counts_from_dataset(dataset):
    counts = Counter()
    for user_id in dataset.valid_users:
        label = dataset.labels_dict.get(user_id)
        if label is None:
            continue
        counts[label['global_ru_id']] += 1
    return counts


def save_plots(output_dir, full_counts, train_counts, val_counts, test_counts, num_classes):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_ids = np.arange(num_classes)

    full_array = np.array([full_counts.get(class_id, 0) for class_id in class_ids])
    train_array = np.array([train_counts.get(class_id, 0) for class_id in class_ids])
    val_array = np.array([val_counts.get(class_id, 0) for class_id in class_ids])
    test_array = np.array([test_counts.get(class_id, 0) for class_id in class_ids])

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].bar(class_ids, full_array, width=0.9, color='#1f77b4')
    axes[0].set_title('Global RU ID distribution in full dataset')
    axes[0].set_xlabel('global_ru_id')
    axes[0].set_ylabel('count')
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].plot(class_ids, train_array, label='train', linewidth=1.5)
    axes[1].plot(class_ids, val_array, label='val', linewidth=1.5)
    axes[1].plot(class_ids, test_array, label='test', linewidth=1.5)
    axes[1].set_title('Global RU ID distribution per split')
    axes[1].set_xlabel('global_ru_id')
    axes[1].set_ylabel('count')
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = output_dir / 'class_imbalance_global_ru_id.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved class imbalance plot to {plot_path}")


def main():
    subthz_path = '../dataset/office_space_inline/sub_thz_channels'
    sub10_path = '../dataset/office_space_inline/sub_10ghz_channels'
    labels_path = '../dataset/office_space_inline/ru_selection_labels/results.csv'

    mode = 'sub10'
    split_seed = 42
    output_dir = Path('runs')

    print('Loading dataset...')
    dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
    )

    num_classes = dataset.num_ru_ids
    full_counts = counts_from_dataset(dataset)

    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    generator = torch.Generator().manual_seed(split_seed)

    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_counts = counts_from_subset(train_subset)
    val_counts = counts_from_subset(val_subset)
    test_counts = counts_from_subset(test_subset)

    print(f"\nMode: {mode}")
    print(f"Dataset size: {total_samples}")
    print(f"Split sizes: train={len(train_subset)}, val={len(val_subset)}, test={len(test_subset)}")

    summarize_counts('Full dataset distribution', full_counts, num_classes)
    summarize_counts('Train split distribution', train_counts, num_classes)
    summarize_counts('Validation split distribution', val_counts, num_classes)
    summarize_counts('Test split distribution', test_counts, num_classes)

    save_plots(output_dir, full_counts, train_counts, val_counts, test_counts, num_classes)


if __name__ == '__main__':
    main()