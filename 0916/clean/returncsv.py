import pandas as pd
import os

# 設定來源資料夾與輸出資料夾
input_folder = r"C:\Users\01254\Desktop\STP\STP\0916\clean"
output_folder = os.path.join(input_folder, "csv_out")

os.makedirs(output_folder, exist_ok=True)

# 找出所有 parquet 檔
for filename in os.listdir(input_folder):
    if filename.endswith(".parquet"):
        parquet_path = os.path.join(input_folder, filename)
        csv_path = os.path.join(output_folder, filename.replace(".parquet", ".csv"))

        # 讀取 parquet
        df = pd.read_parquet(parquet_path, engine="pyarrow")  # 或 "fastparquet"
        
        # 輸出成 CSV
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已轉換 {filename} → {csv_path}")
