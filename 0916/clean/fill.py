import pandas as pd

# 載入檔案
file = r"c:\Users\01254\Desktop\STP\STP\0916\clean\csv_out\20240101.csv"
df = pd.read_csv(file, parse_dates=["time_bin"])

# 建立缺值標記欄位
for col in ["avg_speed", "avg_occupancy"]:
    df[f"is_filled_{col}"] = df[col].isna().astype(int)

# Step 1: 時間序列插值 (只補 NaN)
def time_interpolate(group):
    group = group.set_index("time_bin")
    for col in ["avg_speed", "avg_occupancy"]:
        mask = group[col].isna()
        group.loc[mask, col] = group[col].interpolate(
            method="time", limit_direction="both"
        )[mask]
    return group.reset_index()

df = df.groupby("vd_id", group_keys=False).apply(time_interpolate)

# Step 2: ffill/bfill 補首尾
for col in ["avg_speed", "avg_occupancy"]:
    mask = df[col].isna()
    df.loc[mask, col] = df.groupby("vd_id")[col].ffill().bfill()[mask]

# （這裡暫時不做歷史平均，因為是單日檔案）

# 輸出
output = r"c:\Users\01254\Desktop\STP\STP\0916\clean\csv_out\20240101_filled.csv"
df.to_csv(output, index=False, encoding="utf-8-sig")

print("✅ 單日補值完成：", output)
