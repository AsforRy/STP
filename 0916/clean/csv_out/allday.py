import pandas as pd
import glob
import os

# 檔案路徑（放多天的 CSV）
folder = r"c:\Users\01254\Desktop\STP\STP\0916\clean\csv_out"
files = glob.glob(os.path.join(folder, "*.csv"))

# 讀取所有天資料，建立歷史平均
all_days = []
for f in files:
    df = pd.read_csv(f, parse_dates=["time_bin"])
    df["date"] = df["time_bin"].dt.date
    df["time_of_day"] = df["time_bin"].dt.time
    all_days.append(df)

all_data = pd.concat(all_days, ignore_index=True)

historical_mean = (
    all_data.groupby(["vd_id", "time_of_day"])[["avg_speed", "avg_occupancy"]]
    .mean()
    .reset_index()
)

# 補值函數
def fill_missing(df, historical_mean):
    # 建立缺值標記欄位
    for col in ["avg_speed", "avg_occupancy"]:
        df[f"is_filled_{col}"] = df[col].isna().astype(int)

    # Step 1: 時間序列插值 (只補 NaN)
    df = df.set_index("time_bin")
    for col in ["avg_speed", "avg_occupancy"]:
        mask = df[col].isna()
        df.loc[mask, col] = df[col].interpolate(method="time", limit_direction="both")[mask]
    df = df.reset_index()

    # Step 2: ffill/bfill
    for col in ["avg_speed", "avg_occupancy"]:
        mask = df[col].isna()
        df.loc[mask, col] = df.groupby("vd_id")[col].ffill().bfill()[mask]

    # Step 3: 歷史平均補長缺失
    df["time_of_day"] = df["time_bin"].dt.time
    df = df.merge(
        historical_mean,
        on=["vd_id", "time_of_day"],
        how="left",
        suffixes=("", "_hist"),
    )
    for col in ["avg_speed", "avg_occupancy"]:
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, f"{col}_hist"]
        df.drop(columns=[f"{col}_hist"], inplace=True)

    return df

# 套用到每一天
for f in files:
    df = pd.read_csv(f, parse_dates=["time_bin"])
    df = fill_missing(df, historical_mean)

    output = f.replace(".csv", "_filled.csv")
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print("✅ 補值完成：", output)
