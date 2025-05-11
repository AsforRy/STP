
# 🌸 Iris 資料分析與分類模型範例

本範例以經典的 Iris 花卉資料集，實作資料前處理、探索性資料分析、模型訓練與混淆矩陣評估。

---

## 1️⃣ 資料載入與初步檢查

```python
df = pd.read_csv('Iris.csv')
df = df.drop(columns=["Id"])
df.head()
df.describe()
df.info()
```

---

## 2️⃣ 資料預處理

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.isnull().sum()
```

- LabelEncoder 將花種轉成數字 0, 1, 2
- 檢查缺失值：皆為 0，資料完整

---

## 3️⃣ 探索性資料分析（EDA）

### 🔢 數值分布（直方圖）

```python
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()
```

### 🌈 特徵關係圖（散佈圖）

```python
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=colors)
```

- 多種散佈圖展示花萼、花瓣的幾何特徵

---

## 4️⃣ 特徵相關性分析

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

- 高相關：PetalLengthCm vs PetalWidthCm

---

## 5️⃣ 機器學習模型訓練

```python
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
```

### 🤖 模型建構與訓練

```python
# Logistic Regression
model1 = LogisticRegression()
model1.fit(x_train, y_train)

# K-Nearest Neighbors
model2 = KNeighborsClassifier()
model2.fit(x_train, y_train)

# Decision Tree
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)
```

---

## 6️⃣ 模型評估：準確率

```python
print(model1.score(x_test, y_test))
print(model2.score(x_test, y_test))
print(model3.score(x_test, y_test))
```

---

## 7️⃣ 混淆矩陣（Confusion Matrix）

```python
confusion_matrix(y_test, model.predict(x_test))
sns.heatmap(...)
```

- 視覺化三個模型的預測準確性
- 分析分類錯誤類型與數量

---

## ✅ 模型比較總結（一般情況下的表現）

| 模型                      | 優點                      | 缺點                 |
| ----------------------- | ----------------------- | ------------------ |
| **Logistic Regression** | 計算快，適合線性可分問題；可解釋性強      | 遇到非線性邊界效果差         |
| **K-Nearest Neighbors** | 簡單直觀、非參數式方法；能處理非線性關係    | 大資料慢、對雜訊敏感、不適合高維資料 |
| **Decision Tree**       | 可處理非線性、可視化、能處理類別與數值混合資料 | 容易過擬合，尤其是沒有限制深度時   |

## 🧠 總結建議

| 如果...                | 選用這個模型              |
| -------------------- | ------------------- |
| 你要快速建立基準模型（baseline） | Logistic Regression |
| 你想捕捉非線性關係、資料量不大      | KNN                 |
| 你要可視化分類邏輯、處理複雜邊界     | Decision Tree       |

## 🧠 最佳預測模型：通常是 K-Nearest Neighbors (KNN)

- 原因：
- Iris 資料集的三個花種（Setosa, Versicolor, Virginica）：
- 在特徵空間中 自然分群明顯，邊界非線性。
- 特別是 Setosa 幾乎與其他兩種完全分離，KNN 容易掌握這種分佈。

- KNN 是非參數模型：
- 它不假設特徵與類別之間的線性關係，反而能彈性適應數據分佈。
- 相比 Logistic Regression（只擅長線性邊界）效果通常更好。

- Decision Tree 也會有不錯表現：
- 但若沒有做剪枝（如限制深度、min_samples_leaf），容易過擬合，尤其是在訓練準確率高、測試下降的情況下。

## 📊 模型預測表現（一般經驗值範圍）

| 模型                  | 預測準確率（估計）       |
| ------------------- | --------------- |
| Logistic Regression | 約 93–96%        |
| **KNN**             | ✅ 通常 96–100%    |
| Decision Tree       | 約 93–98%，有時會過擬合 |
