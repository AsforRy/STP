
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

## ✅ 小結

- 資料清洗 → 視覺化 → 模型訓練 → 評估流程完整
- 適合作為機器學習分類入門練習

