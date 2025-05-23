# -*- coding: utf-8 -*-
import pandas as pd # type: ignore
import numpy as np # type: ignore
from ucimlrepo import fetch_ucirepo # type: ignore
import os

os.makedirs("../data", exist_ok=True)

# === 載入資料集 ===
wine_quality = fetch_ucirepo(id=186)
heart_disease = fetch_ucirepo(id=45)

breast_cancer = fetch_ucirepo(id=17)

online_retail = fetch_ucirepo(id=352) 

# === Wine Quality 資料處理 ===
X_wine = wine_quality.data.features.copy()
y_wine = wine_quality.data.targets

# 如果 y 是 DataFrame，則取出第一欄轉成 Series（模型需要一維標籤）
if isinstance(y_wine, pd.DataFrame):
    y_wine = y_wine.iloc[:, 0]

# 轉為二元分類（品質 >= 7 為 1）
y_wine = (y_wine >= 7).astype(int)
y_wine.name = "target"

# 合併特徵與標籤
wine_data = pd.concat([X_wine, y_wine], axis=1)
wine_data.to_csv("../data/wine_processed.csv", index=False)

# === Heart Disease 資料處理 ===
X_heart = heart_disease.data.features.copy()
y_heart = heart_disease.data.targets

# 如果 y 是 DataFrame，則取出第一欄轉成 Series（模型需要一維標籤）
if isinstance(y_heart, pd.DataFrame):
    y_heart = y_heart.iloc[:, 0]

# 轉為二元分類（0=無病，>0=有病）
if y_heart.nunique() > 2:
    y_heart = (y_heart > 0).astype(int)
y_heart.name = "target"

# 合併特徵與標籤
heart_data = pd.concat([X_heart, y_heart], axis=1)

# 處理缺值：將 '?' 轉為 NaN，嘗試轉數值，並刪除無法處理的列
for col in heart_data.columns:
    if heart_data[col].dtype == object:
        heart_data[col] = pd.to_numeric(heart_data[col].replace('?', np.nan), errors='coerce')
heart_data = heart_data.dropna()

# 儲存結果
heart_data.to_csv("../data/heart_processed.csv", index=False)


#Breast Cancer 資料處理#
X_bc = breast_cancer.data.features.copy()
y_bc = breast_cancer.data.targets

# 確保 y 是 Series，並轉為 0/1
if isinstance(y_bc, pd.DataFrame):
    y_bc = y_bc.iloc[:, 0]

# 將 'M' (惡性) 轉為 1，'B' (良性) 轉為 0
y_bc = y_bc.map({'M': 1, 'B': 0})
y_bc.name = "target"

# 合併特徵與標籤
bc_data = pd.concat([X_bc, y_bc], axis=1)
bc_data.to_csv("../data/breast_cancer_processed.csv", index=False)


# === Online Retail ===

X_or = online_retail.data.features.copy()
y_or = online_retail.data.targets 

# 合併特徵與目標
or_data = pd.concat([X_or, y_or], axis=1)

# 去除缺失值
or_data = or_data.dropna() 

# 嘗試找出 Invoice 欄位，並排除退貨（C開頭）
invoice_col = [col for col in or_data.columns if 'invoice' in col.lower()]
if invoice_col:
    col_name = invoice_col[0]
    or_data = or_data[~or_data[col_name].astype(str).str.startswith('C')]

# 建立新欄位：總金額 = 數量 × 單價
or_data['TotalAmount'] = or_data['Quantity'] * or_data['UnitPrice']

# 建立二元分類目標欄位：是否為大額訂單（總金額 > 100）
or_data['target'] = (or_data['TotalAmount'] > 100).astype(int)

# 保留必要欄位
or_data = or_data[['Quantity', 'UnitPrice', 'TotalAmount', 'Country', 'target']]

# 對類別欄位做 one-hot 編碼
or_data = pd.get_dummies(or_data, columns=['Country'])

# 儲存結果
or_data.to_csv("../data/online_retail_processed.csv", index=False)

print("✅ 預處理完成，已儲存 wine_processed.csv 與 heart_processed.csv 與 breast_cancer_processed.csv and online_retail_processed.csv")


"""
| 結構          | 維度 | 像什麼      | 用途                 |
| ----------- | -- | -------- | ------------------ |
| `Series`    | 1D | 一欄數據     | 機器學習的 y（標籤、目標）     |
| `DataFrame` | 2D | 表格（多欄數據） | X 特徵資料、整份資料集的儲存與處理 |
"""

