from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as exc:
    missing = exc.name or "matplotlib/seaborn"
    raise SystemExit(
        "Missing plotting dependency. Run with:\n"
        "  uv run --with matplotlib --with seaborn python -m experiments.plot_tx_rx_delta_regression_table"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = ROOT / "experiments" / "results" / "tx_rx_delta_regression.md"
OUTPUT_PATH = ROOT / "experiments" / "results" / "tx_rx_delta_regression_plot.png"


@dataclass(frozen=True)
class ResultRow:
    label: str
    group: str
    rmse_sklearn: float
    acc_2m: float


def _clean_cell(value: str) -> str:
    return value.strip().replace("**", "").replace("`", "")


def _split_markdown_row(line: str) -> list[str]:
    return [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]


def _parse_markdown_tables(text: str) -> list[list[dict[str, str]]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    tables: list[list[dict[str, str]]] = []
    for block in blocks:
        header = _split_markdown_row(block[0])
        rows: list[dict[str, str]] = []
        for line in block[2:]:
            values = _split_markdown_row(line)
            if len(values) != len(header):
                continue
            rows.append(dict(zip(header, values, strict=True)))
        tables.append(rows)

    return tables


def _parse_float(value: str) -> float:
    return float(value.replace("%", "").strip())


def _short_name(model_name: str) -> str:
    mapping = {
        "Global multi-output": "Global",
        "ExtraTrees": "ET",
        "RandomForest": "RF",
        "MLP": "MLP",
        "XGBoost": "XGB",
    }
    return mapping.get(model_name, model_name)


def load_rows() -> list[ResultRow]:
    tables = _parse_markdown_tables(MARKDOWN_PATH.read_text())
    if len(tables) < 2:
        raise ValueError(f"Expected at least two markdown tables in {MARKDOWN_PATH}")

    rows: list[ResultRow] = []

    for row in tables[0]:
        rows.append(
            ResultRow(
                label="Global",
                group="global",
                rmse_sklearn=_parse_float(row["RMSE sklearn (m)"]),
                acc_2m=_parse_float(row["acc@2m"]),
            )
        )

    for mode, group, suffix in [
        ("All features", "all_features", "all"),
        ("Per-Tx features", "per_tx_features", "per-tx"),
    ]:
        for row in tables[1]:
            if row["Feature mode"] != mode:
                continue
            rows.append(
                ResultRow(
                    label=f"{_short_name(row['Model'])} {suffix}",
                    group=group,
                    rmse_sklearn=_parse_float(row["RMSE sklearn (m)"]),
                    acc_2m=_parse_float(row["acc@2m"]),
                )
            )

    return rows


def _palette(rows: list[ResultRow]) -> dict[str, str]:
    return {
        row.label: {
            "global": "#334155",
            "all_features": "#2563eb",
            "per_tx_features": "#d97706",
        }[row.group]
        for row in rows
    }


def _annotate_bars(ax, values: list[float], percent: bool = False) -> None:
    max_value = max(values)
    offset = max_value * 0.015
    for patch, value in zip(ax.patches, values, strict=True):
        label = f"{value:.1f}%" if percent else f"{value:.3f}"
        ax.text(
            patch.get_width() + offset,
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=10,
            color="#0f172a",
        )


def plot_rows(rows: list[ResultRow]) -> Path:
    sns.set_theme(style="whitegrid")

    labels = [row.label for row in rows]
    rmse_values = [row.rmse_sklearn for row in rows]
    acc_values = [row.acc_2m for row in rows]
    palette = _palette(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    fig.suptitle("Tx/Rx delta regression from markdown table", fontsize=18, fontweight="bold")

    sns.barplot(
        ax=axes[0],
        x=rmse_values,
        y=labels,
        hue=labels,
        palette=palette,
        dodge=False,
        legend=False,
        orient="h",
    )
    axes[0].set_title("RMSE sklearn")
    axes[0].set_xlabel("meters")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, max(rmse_values) * 1.18)
    _annotate_bars(axes[0], rmse_values)

    sns.barplot(
        ax=axes[1],
        x=acc_values,
        y=labels,
        hue=labels,
        palette=palette,
        dodge=False,
        legend=False,
        orient="h",
    )
    axes[1].set_title("acc@2m")
    axes[1].set_xlabel("percent")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 100)
    _annotate_bars(axes[1], acc_values, percent=True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in ["#334155", "#2563eb", "#d97706"]
    ]
    labels = ["global", "all_features", "per_tx_features"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    output_path = plot_rows(load_rows())
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
