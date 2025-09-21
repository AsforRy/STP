import pandas as pd

parquet_file = r"c:\Users\01254\Desktop\STP\STP\0916\clean\20240101.parquet"
csv_file = r"c:\Users\01254\Desktop\STP\STP\0916\clean\csv_out\20240101.csv"

df_parquet = pd.read_parquet(parquet_file, engine="pyarrow")
df_csv = pd.read_csv(csv_file)

print("Parquet 筆數:", len(df_parquet))
print("CSV 筆數:", len(df_csv))

# 檢查前 5 筆是否一樣
print("Parquet 前5筆：")
print(df_parquet.head())
print("CSV 前5筆：")
print(df_csv.head())
