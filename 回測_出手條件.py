import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 讀取資料
df = pd.read_csv("文章關鍵字向量.csv", encoding="utf-8-sig")
df['post_time'] = pd.to_datetime(df['post_time'])
df = df[df['標記'].isin([-1, 1])].reset_index(drop=True)
df['date'] = df['post_time'].dt.date

# 建立每篇文章的 D+5 標籤
df['D+5_label'] = df['標記'].shift(-5)
df = df.dropna(subset=['D+5_label']).copy()
df['D+5_label'] = df['D+5_label'].astype(int)
df = df[df['D+5_label'].isin([-1, 1])].reset_index(drop=True)

# 建立訓練與測試資料
df['post_time'] = pd.to_datetime(df['post_time'])
df['date'] = df['post_time'].dt.date
unique_dates = sorted(df['date'].unique())

results = []
window_days = 60
confidence_threshold = 0.7

for idx in range(window_days, len(unique_dates) - 5):
    train_start = unique_dates[idx - window_days]
    train_end = unique_dates[idx - 1]
    test_day = unique_dates[idx]

    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test_df = df[df['date'] == test_day]

    if len(train_df) < 10 or len(set(train_df['D+5_label'])) < 2 or len(test_df) == 0:
        continue

    X_train = train_df.drop(columns=['文章ID', 'post_time', '標記', 'D+5_label', 'month', 'date'], errors='ignore')
    y_train = train_df['D+5_label']
    X_test = test_df.drop(columns=['文章ID', 'post_time', '標記', 'D+5_label', 'month', 'date'], errors='ignore')
    y_test = test_df['D+5_label']

    model = LogisticRegression(class_weight='balanced', max_iter=10000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    confidence = np.max(probs, axis=1)
    pred = model.predict(X_test)

    test_df = test_df.copy()
    test_df['預測'] = pred
    test_df['實際'] = y_test.values
    test_df['信心'] = confidence
    test_df['信心足夠'] = test_df['信心'] >= confidence_threshold

    results.append(test_df[['post_time', '預測', '實際', '信心', '信心足夠']])

# 結合所有結果
df_result = pd.concat(results).reset_index(drop=True)
df_out = df_result[df_result['信心足夠']]
total_articles = len(df_result)
shot_articles = len(df_out)
hit_articles = (df_out['預測'] == df_out['實際']).sum()
hit_rate = hit_articles / shot_articles if shot_articles > 0 else 0
shot_rate = shot_articles / total_articles if total_articles > 0 else 0

print("\n✅ 每篇文章預測 D+5 結果")
print(f"出手率：{shot_rate:.2%}")
print(f"準確率（出手時）：{hit_rate:.2%}")

# 混淆矩陣
if shot_articles > 0:
    y_true = df_out['實際']
    y_pred = df_out['預測']
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    print("\n Confusion Matrix（以文章為單位）")
    print("           預測為漲   預測為跌")
    print(f"實際為漲     {cm[0][0]}        {cm[0][1]}")
    print(f"實際為跌     {cm[1][0]}        {cm[1][1]}")

import matplotlib.pyplot as plt

# 月份欄位
df_out['Month'] = pd.to_datetime(df_out['post_time']).dt.to_period('M').astype(str)

# 每月命中率
monthly_stats = df_out.copy()
monthly_stats['Hit'] = (monthly_stats['預測'] == monthly_stats['實際']).astype(int)
monthly_hit_rate = monthly_stats.groupby('Month')['Hit'].mean().reset_index()

# 用整體 hit_rate 當作平均線
plt.figure(figsize=(10, 5))
plt.plot(monthly_hit_rate['Month'], monthly_hit_rate['Hit'], marker='o', linestyle='-', color='blue', label='Monthly Hit Rate')
plt.axhline(hit_rate, color='red', linestyle='--', label=f'Overall Hit Rate: {hit_rate:.2%}')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.ylabel('Hit Rate')
plt.legend()
plt.tight_layout()
plt.savefig("monthly_hit_rate_linechart.png", dpi=300)
plt.show()


