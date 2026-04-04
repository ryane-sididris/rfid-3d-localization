"""
Raw RSSI loader (AEP_OK).

Reads raw .txt files from raw_data/ and returns a long-format DataFrame
with one row per (timestamp, Tx, rx) reading and x, y, z labels.

Columns: conf, run, datestamp, EPC, Tx, rx, txPower_dBm, rssi_dBm,
         rssi_linear, x, y, z
"""

import json
from pathlib import Path

import pandas as pd

INVALID_RSSI_DBM = -100.0


def load(raw_root: Path | str) -> pd.DataFrame:
    """Load all AEP_OK runs under *raw_root* into a long-format DataFrame."""
    raw_root = Path(raw_root)
    tags_raw, roundstart_raw, runs_df = _load_raw_data(raw_root)
    roundstart = _expand_roundstart(roundstart_raw)

    tags = tags_raw.merge(roundstart, on=["conf", "run", "round"], how="left")
    tags["EPC"] = tags["data"].apply(_extract_epc)
    tags["datestamp"] = pd.to_datetime(tags["datestamp"], format="%y/%m/%d %H:%M:%S.%f")
    tags["rssi1_dBm"] = tags["rssi"].str[0].astype(float)
    tags["rssi2_dBm"] = tags["rssi"].str[1].astype(float)

    tags_long = pd.melt(
        tags,
        id_vars=["conf", "run", "datestamp", "EPC", "Tx", "rx1_ant", "rx2_ant", "rx1_rx2_ant", "txPower_dBm"],
        value_vars=["rssi1_dBm", "rssi2_dBm"],
        var_name="rssi_slot",
        value_name="rssi_dBm",
    )
    tags_long.loc[tags_long["rssi_slot"] == "rssi1_dBm", "rx"] = tags_long["rx1_ant"]
    tags_long.loc[tags_long["rssi_slot"] == "rssi2_dBm", "rx"] = tags_long["rx2_ant"]
    tags_long = tags_long[tags_long["rssi_dBm"] > INVALID_RSSI_DBM].copy()
    tags_long["rssi_linear"] = 1e6 * 10 ** (tags_long["rssi_dBm"] / 10)
    tags_long = tags_long.drop(columns=["rx1_ant", "rx2_ant", "rssi_slot"])

    actuals = _load_actuals(raw_root)
    runs_small = runs_df[["conf", "run", "x", "y", "z0"]].drop_duplicates()
    actuals_runs = actuals.merge(runs_small, on=["conf"])
    actuals_runs["z"] = actuals_runs["z0"] + actuals_runs["z_rel"]
    actuals_runs = actuals_runs[["conf", "run", "EPC", "x", "y", "z"]]

    dataset = tags_long.merge(actuals_runs, on=["conf", "run", "EPC"], how="inner")
    dataset = (
        dataset[["conf", "run", "datestamp", "EPC", "Tx", "rx", "txPower_dBm", "rssi_dBm", "rssi_linear", "x", "y", "z"]]
        .sort_values(["run", "datestamp", "Tx", "rx"])
        .reset_index(drop=True)
    )
    return dataset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_datestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%y/%m/%d %H:%M:%S.%f")


def _parse_file_metadata(file_path: Path) -> dict:
    parts = file_path.parts
    idx = parts.index("raw_data")
    tokens = {}
    for token in file_path.stem.split("__"):
        if "=" in token:
            key, val = token.split("=", 1)
            tokens[key] = val
    return {
        "conf": parts[idx + 1],
        "x": float(tokens["x"]),
        "y": float(tokens["y"]),
        "z0": float(tokens["z0"]),
    }


def _parse_txt_file(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = _parse_file_metadata(file_path)
    tags_rows, round_rows = [], []
    run_ts = None
    with file_path.open("rt") as handle:
        for line in handle:
            if not line.startswith("data: "):
                continue
            try:
                data_json = json.loads(line.split("data: ", 1)[1])
            except json.JSONDecodeError:
                continue  # skip truncated / malformed lines
            if run_ts is None:
                run_ts = _parse_datestamp(data_json["datestamp"]).round("1s")
            data_type = data_json.get("type")
            if data_type == "TagReadData":
                tags_rows.append(data_json)
            elif data_type == "RoundStart":
                round_rows.append(data_json)

    tags_df = pd.DataFrame(tags_rows)
    round_df = pd.DataFrame(round_rows)
    tags_df["run"] = run_ts
    round_df["run"] = run_ts
    tags_df["conf"] = meta["conf"]
    round_df["conf"] = meta["conf"]
    run_info = {"conf": meta["conf"], "run": run_ts, "x": meta["x"], "y": meta["y"], "z0": meta["z0"]}
    return tags_df, round_df, run_info


def _load_raw_data(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tag_frames, round_frames, runs = [], [], []
    for file_path in sorted(raw_root.rglob("*.txt")):
        if "pgm=AEP_OK" not in file_path.parts:
            continue
        tags_df, round_df, run_info = _parse_txt_file(file_path)
        tag_frames.append(tags_df)
        round_frames.append(round_df)
        runs.append(run_info)
    if not tag_frames:
        raise FileNotFoundError(f"No AEP_OK .txt files found under {raw_root}")
    return (
        pd.concat(tag_frames, ignore_index=True),
        pd.concat(round_frames, ignore_index=True),
        pd.DataFrame(runs).drop_duplicates(),
    )


def _expand_roundstart(roundstart: pd.DataFrame) -> pd.DataFrame:
    def rx_port(cfg: dict, idx: int) -> str:
        return f"{cfg[f'antennaPort{idx}']}__{cfg[f'expanderPort{idx}']}"

    rs = roundstart.copy()
    rs["Tx"] = rs[["txAntennaPort", "txExpanderPort"]].apply(lambda x: "__".join(x), axis=1)
    rs["rx1_ant"] = rs["rxAntennaConfig"].apply(lambda x: rx_port(x, 1))
    rs["rx2_ant"] = rs["rxAntennaConfig"].apply(lambda x: rx_port(x, 2))
    rs["rx1_rx2_ant"] = rs["rx1_ant"] + "|" + rs["rx2_ant"]
    return rs[["conf", "run", "round", "Tx", "rx1_ant", "rx2_ant", "rx1_rx2_ant", "txPower_dBm", "freq_MHz"]]


def _extract_epc(value: str) -> str:
    return value[6:30]


def _load_actuals(raw_root: Path) -> pd.DataFrame:
    frames = []
    for conf_dir in raw_root.iterdir():
        if not conf_dir.is_dir():
            continue
        actuals_dir = conf_dir / "Actuals"
        if not actuals_dir.exists():
            continue
        for csv_path in actuals_dir.glob("*.csv"):
            df = pd.read_csv(csv_path, sep=";", decimal=",", dtype=str)
            df = df[["EPC", "z_rel"]].copy()
            df["conf"] = conf_dir.name
            frames.append(df)
    actuals = pd.concat(frames, ignore_index=True)
    actuals["z_rel"] = actuals["z_rel"].astype(str).str.replace(",", ".").astype(float)
    return actuals
