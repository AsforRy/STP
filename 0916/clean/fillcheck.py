import pandas as pd
import matplotlib.pyplot as plt

# 載入原始與補值後檔案
raw_file = r"c:\Users\01254\Desktop\STP\STP\0916\clean\csv_out\20240101.csv"
filled_file = r"c:\Users\01254\Desktop\STP\STP\0916\clean\csv_out\20240101_filled.csv"

df_raw = pd.read_csv(raw_file, parse_dates=["time_bin"])
df_filled = pd.read_csv(filled_file, parse_dates=["time_bin"])

# 1️⃣ 統計檢查
print("===== 統計檢查 =====")
for col in ["avg_speed", "avg_occupancy"]:
    print(f"\n【{col}】")
    print("NaN 數量： 原始 =", df_raw[col].isna().sum(), ", 補值後 =", df_filled[col].isna().sum())
    print("描述統計 (原始):")
    print(df_raw[col].describe())
    print("描述統計 (補值後):")
    print(df_filled[col].describe())

# 2️⃣ 確定哪些地方有補到缺值
print("\n===== 補值檢查 =====")
for col in ["avg_speed", "avg_occupancy"]:
    fixed_rows = df_raw[col].isna() & df_filled[col].notna()
    print(f"\n【{col}】補上的筆數:", fixed_rows.sum())
    if fixed_rows.sum() > 0:
        print("示例：")
        print(df_filled.loc[fixed_rows, ["time_bin", "vd_id", col]].head(10))

# 3️⃣ 邏輯檢查 (異常值篩選)
print("\n===== 邏輯檢查 =====")
print("補值後 avg_speed <0 或 >200：", df_filled[(df_filled["avg_speed"] < 0) | (df_filled["avg_speed"] > 200)].shape[0])
print("補值後 avg_occupancy <0 或 >100：", df_filled[(df_filled["avg_occupancy"] < 0) | (df_filled["avg_occupancy"] > 100)].shape[0])

# 4️⃣ 視覺化檢查 (隨機挑選幾個 vd_id)
sample_vds = df_raw["vd_id"].dropna().unique()[:3]  # 挑 3 個感測器
for vd in sample_vds:
    raw_subset = df_raw[df_raw["vd_id"] == vd]
    filled_subset = df_filled[df_filled["vd_id"] == vd]

    plt.figure(figsize=(12,5))
    plt.plot(raw_subset["time_bin"], raw_subset["avg_speed"], label="ori avg_speed", alpha=0.7, marker="o")
    plt.plot(filled_subset["time_bin"], filled_subset["avg_speed"], label="filled avg_speed", alpha=0.7)
    plt.title(f"vd_id={vd} 缺值補齊比較")
    plt.xlabel("time")
    plt.ylabel("speed (km/h)")
    plt.legend()
    plt.show()
