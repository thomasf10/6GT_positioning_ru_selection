"""
Quick check: do sub_10ghz_channels/ and sub_thz_channels/ contain the exact
same set of ue_*.nc user IDs for a given dataset folder?

If not, the train_csi_ru_beam_sel.py (mode='sub10') and
train_subthz_csi_ru_beam_sel.py (mode='subTHz') scripts will build
different `valid_users` lists, leading to different train/val/test splits
even with the same random seed.
"""
import os
import re
from pathlib import Path


def extract_ue_ids(folder):
    ids = set()
    if not os.path.isdir(folder):
        print(f"  [missing folder] {folder}")
        return ids
    for f in os.listdir(folder):
        m = re.search(r'ue_(\d+)', f)
        if m:
            ids.add(int(m.group(1)))
    return ids


def compare(dataset_folder):
    root = Path(__file__).parent / dataset_folder
    sub10 = root / 'sub_10ghz_channels'
    subthz = root / 'sub_thz_channels'

    sub10_ids = extract_ue_ids(sub10)
    subthz_ids = extract_ue_ids(subthz)

    only_sub10 = sorted(sub10_ids - subthz_ids)
    only_subthz = sorted(subthz_ids - sub10_ids)
    common = sub10_ids & subthz_ids

    print(f"\n=== {dataset_folder} ===")
    print(f"  sub_10ghz_channels:  {len(sub10_ids)} UEs")
    print(f"  sub_thz_channels:    {len(subthz_ids)} UEs")
    print(f"  intersection:        {len(common)} UEs")
    print(f"  only in sub_10ghz:   {len(only_sub10)}  {only_sub10[:20]}{' ...' if len(only_sub10) > 20 else ''}")
    print(f"  only in sub_thz:     {len(only_subthz)}  {only_subthz[:20]}{' ...' if len(only_subthz) > 20 else ''}")

    if not only_sub10 and not only_subthz:
        print("  OK: folder UE sets are identical.")
    else:
        print("  MISMATCH: sets differ — this will cause different train/val/test splits.")


if __name__ == '__main__':
    compare('office_space_reduced_inline')
