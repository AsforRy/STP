# plot_results.py
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os

def plot_comparison_charts(csv_path="../results/comparison_results.csv", output_dir="../results"):
    # 讀取比較結果
    df = pd.read_csv(csv_path)

    # 設定繪圖樣式
    sns.set(style="whitegrid")

    # === F1 Score 圖表 ===
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="F1 Score", hue="Dataset")
    plt.xticks(rotation=45)
    plt.title("模型在不同資料集上的 F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"))
    plt.close()

    # === Accuracy 圖表 ===
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Dataset")
    plt.xticks(rotation=45)
    plt.title("模型在不同資料集上的 Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

    print("📈 圖表已儲存至 results/ 資料夾")
