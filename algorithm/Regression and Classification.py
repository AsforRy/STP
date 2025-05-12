import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# 設定隨機種子
random_state=None

# 🔧 產生 200 筆假房屋資料
n = 200
area = np.random.randint(20, 150, size=n)  # 面積：20~150 平方米
age = np.random.randint(0, 30, size=n)     # 房齡：0~30 年
distance_to_mrt = np.random.randint(50, 1000, size=n)  # 距捷運距離：50~1000 公尺
has_parking = np.random.choice([0, 1], size=n)         # 是否有車位

# 🔢 價格的生成公式（模擬現實邏輯）
# 基礎價格與面積成正比，房齡、距離越大價格越低，有車位加值
price = (area * 15) - (age * 10) - (distance_to_mrt * 0.5) + (has_parking * 500) + np.random.normal(0, 100, size=n)
price = np.round(price).astype(int)

# 🔤 標籤：是否高價房（定義 > 1500 萬為高價）
is_high_price = price > 1500

# ✅ 組成 DataFrame
data = pd.DataFrame({
    'area': area,
    'age': age,
    'distance_to_mrt': distance_to_mrt,
    'has_parking': has_parking,
    'price': price,
    'is_high_price': is_high_price
})

# 🎯 特徵與目標
X = data[['area', 'age', 'distance_to_mrt', 'has_parking']]
y_reg = data['price']
y_cls = data['is_high_price']

# 📊 分割資料
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.3, random_state=42
)

# 🔵 回歸模型
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
reg_pred = reg_model.predict(X_test)
print("✅ 回歸 MSE（預測房價）:", round(mean_squared_error(y_reg_test, reg_pred), 2))

# 🔴 分類模型
cls_model = LogisticRegression()
cls_model.fit(X_train, y_cls_train)
cls_pred = cls_model.predict(X_test)
print("✅ 分類準確率（高價 vs 低價）:", round(accuracy_score(y_cls_test, cls_pred), 2))
