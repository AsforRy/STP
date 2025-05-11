#Import Required Libraries

#匯入 pandas 套件，常用於讀取與處理表格型資料(ex : csv)
import pandas as pd
#匯入 NumPy，提供支援陣列、數值運算、統計等功能
import numpy as np
#匯入 os 模組，讓你能操作系統路徑、檔案、目錄等
import os
#匯入 matplotlib 的 pyplot 模組，通常用來畫圖或資料視覺化
import matplotlib.pyplot as plt
#匯入 seaborn，它是建立在 matplotlib 之上的統計視覺化套件，風格較美觀
import seaborn as sns
#關閉所有警告訊息
import warnings
warnings.filterwarnings('ignore')

#Load the dataset

#使用 pandas 讀取名為 Iris.csv 的 CSV 檔案，並存入 df（dataframe）變數中
df = pd.read_csv('Iris.csv')
#顯示資料表（dataframe）前 5 筆資料，方便快速瀏覽欄位與數據格式
df.head()

#移除資料表中不需要的 Id 欄位，因為這個欄位只是編號，不影響分析
df = df.drop(columns=["Id"])
#再次顯示前 5 筆資料（這時 Id 欄位已經移除了）
df.head()

#顯示資料表中 前 10 筆資料，可以看到更多花卉樣本的詳細數據
df.head(10)

"""
使用 pandas 的 describe() 方法，顯示數值型欄位的基本統計資料，包括：
count：非空值的數量
mean：平均值
std：標準差（資料離散程度）
min：最小值
25%、50%（中位數）、75%：分位數（描述資料分布）
max：最大值
通常用來快速了解資料的範圍與分布
"""
df.describe()

""""
使用 info() 方法來顯示資料表的整體結構資訊，包括：
每個欄位的名稱與順序
欄位的資料型態（例如 float64, int64, object 等）
非空值的筆數（可用來判斷是否有缺漏資料）
記憶體使用量
非常適合用來檢查是否有 缺失值 或 類別資料
"""
df.info()

#顯示 Species 欄位中每個類別（花卉品種）出現的次數
df['Species'].value_counts()

#Preprocessing the Dataset

#從 scikit-learn 的 preprocessing 模組中匯入 LabelEncoder，這個類別可以把「字串類別」轉換成「數值編號」
from sklearn.preprocessing import LabelEncoder
#建立一個 LabelEncoder 的實例（物件），命名為 le，準備對類別欄位進行轉換
le = LabelEncoder()
#對 Species 欄位的每個類別進行轉換，將字串類別變成整數編號，並把結果存回 df['Species']
df['Species'] = le.fit_transform(df['Species'])
#顯示轉換後的 Species 欄位內容，也就是每筆資料現在都變成數字 0、1、2 的形式
df['Species']

#顯示整個 DataFrame（df），此時 Species 欄位已經從字串變成數字，方便模型訓練
df

"""
檢查資料是否有缺失值
df.isnull()
對整個 DataFrame 執行「是否為缺失值」的判斷，會回傳一個布林值表格（True 代表該格是 null）
.sum()
對每個欄位將 True 數量加總（因為 True 被視為 1，False 為 0），得到每一欄中「缺失值的數量」
"""
df.isnull().sum()

#Exploratory Data Analysis (EDA)

#從資料表中選出「花萼長度」這欄，並畫出這個欄位的值的分布圖
df['SepalLengthCm'].hist()
#畫出花萼寬度這一欄的數值分布情形
df['SepalWidthCm'].hist()
#顯示花瓣長度的數值分布圖
df['PetalLengthCm'].hist()
#展示花瓣寬度的直方圖，觀察其數值分布
df['PetalWidthCm'].hist()

#畫散佈圖來觀察特徵之間的關係
#對應三個花種的資料點將以red、orange、blue三種顏色畫出
colors = ['red', 'orange', 'blue']
#species 是對應已經經過 LabelEncoder 編碼後的花種標籤（0：Setosa、1：Versicolor、2：Virginica）
species = [0, 1, 2]

#花萼長 vs 花萼寬
#一次處理一種花種的資料
for i in range(3):
    #選出第 i 種花，把這些資料存成新的變數 x
    x = df[df['Species'] == species[i]]
    #用 plt.scatter() 畫出這類花的花萼長度（x 軸）和花萼寬度（y 軸）關係圖
    #c=colors[i]：這類花用第 i 種顏色畫
    #label=species[i]：給圖例（legend）加上標籤，例如 0、1、2
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
#設定 x 軸的標籤為 "Sepal Length"
plt.xlabel("Sepal Length")
#設定 y 軸的標籤為 "Sepal Width"
plt.ylabel("Sepal Width")
#顯示圖例，圖例會對應 label=species[i] 的內容（也就是 0、1、2）
plt.legend()

#花瓣長 vs 花瓣寬
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()

#花萼長 vs 花瓣長
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()

#花萼寬 vs 花瓣寬
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()

#Correlation Matrix

"""
df.corr() 會計算資料集中所有數值欄位之間的 皮爾森相關係數 (Pearson correlation coefficient)
回傳的是一個方陣（DataFrame），代表任兩欄之間的線性相關程度： 1 表示完全正相關
                                                        -1 表示完全負相關
                                                        0 表示無相關
"""
df.corr()

# display the correlation matrix using a heatmap
#再次執行 df.corr()，並把結果存到 corr 變數中，以便後續使用（畫圖）
corr = df.corr()
#建立一個新的圖表與子圖（figure 和 axes） ex: figsize=(5, 4) 設定圖表的寬高為 5 吋 x 4 吋
fig, ax = plt.subplots(figsize=(5, 4))
"""
使用 seaborn 繪製熱力圖（heatmap）來呈現相關係數矩陣：
corr：要視覺化的相關係數資料
annot=True：在每個格子內標示出數值
ax=ax：指定繪圖的目標子圖（前一行定義的 ax）
cmap='coolwarm'：使用紅藍配色（紅為正相關，藍為負相關，中間為白）
"""
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')

#Model Training

# Split the dataset into training and testing sets
#從 sklearn 匯入 train_test_split 函數，用來將資料集拆成訓練集與測試集
from sklearn.model_selection import train_test_split
#將 Species（目標欄位）以外的欄位作為特徵（X）輸入模型
X = df.drop(columns=['Species'])
#將 Species 欄作為標籤（Y），即模型要學會預測的對象
Y = df['Species']
"""
將資料分成 60% 訓練、40% 測試：
x_train, y_train: 訓練資料
x_test, y_test: 測試資料
（test_size=0.40 表示保留 40% 給測試）
"""
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.40)

# Logistic Regression Model(邏輯回歸)
#匯入邏輯回歸分類模型
from sklearn.linear_model import LogisticRegression
#建立一個邏輯回歸模型的實例 model1
model1 = LogisticRegression()
#使用訓練資料 x_train, y_train 來訓練模型
model1.fit(x_train, y_train)
#在測試資料上評估準確率，乘以 100 變成百分比格式顯示
print("Accuracy (Logistic Regression): ", model1.score(x_test, y_test) * 100)

# K-nearest Neighbours Model (KNN)(K最近鄰分類)
#匯入 KNN 分類模型
from sklearn.neighbors import KNeighborsClassifier
#建立一個 KNN 模型的實例 model2（預設 k=5）
model2 = KNeighborsClassifier()
#用訓練資料訓練模型
model2.fit(x_train, y_train)
#測試模型在測試資料的分類準確度
print("Accuracy (KNN): ", model2.score(x_test, y_test) * 100)

# Decision Tree Model(決策樹分類)
#匯入決策樹分類器
from sklearn.tree import DecisionTreeClassifier
#建立決策樹模型實例 model3
model3 = DecisionTreeClassifier()
#用訓練資料訓練模型
model3.fit(x_train, y_train)
#顯示測試資料的分類準確度（百分比格式）
print("Accuracy (Decision Tree): ", model3.score(x_test, y_test) * 100)

#Confusion Matrix(混淆矩陣)

#從 sklearn.metrics 匯入 confusion_matrix 函數，用來計算分類模型的預測結果與真實標籤之間的對照
from sklearn.metrics import confusion_matrix

#使用邏輯回歸模型 (model1) 對測試資料進行預測，結果存到 y_pred1
y_pred1 = model1.predict(x_test)
#使用 KNN 模型進行預測
y_pred2 = model2.predict(x_test)
#使用決策樹模型進行預測
y_pred3 = model3.predict(x_test)

#比較 Logistic Regression 的預測值與實際值，建立混淆矩陣
conf_matrix1 = confusion_matrix(y_test, y_pred1)
#建立 KNN 模型的混淆矩陣
conf_matrix2 = confusion_matrix(y_test, y_pred2)
#建立 Decision Tree 模型的混淆矩陣
conf_matrix3 = confusion_matrix(y_test, y_pred3)

#Logistic Regression 的混淆矩陣視覺化
#建立一個新圖表，設定大小為 8x6 吋
plt.figure(figsize=(8, 6))
"""使用 seaborn 繪製混淆矩陣的熱力圖：
annot=True：在格子內顯示數值
fmt='d'：數值格式為整數
cmap='Blues'：使用藍色調色盤
xticklabels, yticklabels：標籤對應類別（0,1,2）
"""
sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
#設定 X/Y 軸與圖標標題，並顯示圖表
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Logistic Regression')
plt.show()

#KNN 的混淆矩陣視覺化
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of KNN')
plt.show()

#Decision Tree 的混淆矩陣視覺化
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix3, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Decision Tree')
plt.show()