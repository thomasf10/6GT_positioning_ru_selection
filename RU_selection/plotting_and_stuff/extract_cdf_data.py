"""Extract CDF/NMSE data from cdf_nmse_fixed_beams.tex.

Parses every non-commented `\\addplot ... table {% ... };` block in the tikz
file and pairs it with the `\\addlegendentry{...}` that follows. Data is
written as one CSV per plot (columns: nmse_db, cdf) plus a combined CSV
with a `label` column.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

TEX_FILE = Path(__file__).with_name("cdf_nmse_fixed_beams.tex")
OUT_DIR = Path(__file__).with_name("extracted_data")


def strip_comment(line: str) -> str | None:
    """Return the non-comment portion of a line, or None if fully commented."""
    stripped = line.lstrip()
    if stripped.startswith("%"):
        return None
    # Remove trailing inline comment (unescaped %).
    out = []
    prev = ""
    for ch in line:
        if ch == "%" and prev != "\\":
            break
        out.append(ch)
        prev = ch
    return "".join(out)


def sanitize_filename(label: str) -> str:
    # Keep LaTeX macro names (drop just the backslash) so `\ghz` vs `\thz` survive.
    label = re.sub(r"\\([A-Za-z]+)", r"\1", label)
    label = re.sub(r"[^\w\-]+", "_", label, flags=re.UNICODE).strip("_")
    return label or "plot"


def parse_plots(tex_path: Path) -> list[dict]:
    plots: list[dict] = []
    lines = tex_path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        active = strip_comment(lines[i])
        if active is None or "\\addplot" not in active:
            i += 1
            continue

        style = active.strip()

        # Advance to the `table {%` opener (still in the same plot block).
        i += 1
        while i < len(lines):
            live = strip_comment(lines[i])
            if live is not None and "table" in live and "{" in live:
                break
            i += 1
        i += 1  # step past the `table {%` line

        data: list[tuple[float, float]] = []
        while i < len(lines):
            live = strip_comment(lines[i])
            if live is None:
                i += 1
                continue
            if "};" in live:
                i += 1
                break
            parts = live.split()
            if len(parts) >= 2:
                try:
                    data.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            i += 1

        # Look ahead for the matching \addlegendentry (skip commented lines).
        label = ""
        j = i
        while j < len(lines):
            live = strip_comment(lines[j])
            if live is None:
                j += 1
                continue
            m = re.search(r"\\addlegendentry\{(.+?)\}", live)
            if m:
                label = m.group(1).strip()
                break
            if "\\addplot" in live:
                break
            j += 1

        plots.append({"label": label, "style": style, "data": data})

    return plots


def main() -> None:
    plots = parse_plots(TEX_FILE)
    OUT_DIR.mkdir(exist_ok=True)

    combined = OUT_DIR / "all_plots.csv"
    with combined.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "nmse_db", "cdf"])
        for p in plots:
            name = sanitize_filename(p["label"])
            per_plot = OUT_DIR / f"{name}.csv"
            with per_plot.open("w", newline="", encoding="utf-8") as pf:
                pw = csv.writer(pf)
                pw.writerow(["nmse_db", "cdf"])
                for x, y in p["data"]:
                    pw.writerow([x, y])
                    w.writerow([p["label"], x, y])
            print(f"{p['label']!r}: {len(p['data'])} points -> {per_plot.name}")

    print(f"\nCombined CSV: {combined}")


if __name__ == "__main__":
    main()
