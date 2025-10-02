import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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

# 時間列表
unique_dates = sorted(df['date'].unique())
results = []

# 參數
window_days = 60

for idx in range(window_days, len(unique_dates) - 5):
    train_start = unique_dates[idx - window_days]
    train_end = unique_dates[idx - 1]
    test_day = unique_dates[idx]

    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test_df = df[df['date'] == test_day]

    if len(train_df) == 0 or len(set(train_df['D+5_label'])) < 2 or len(test_df) == 0:
        continue

    X_train = train_df.drop(columns=['文章ID', 'post_time', '標記', 'D+5_label', 'month', 'date'], errors='ignore')
    y_train = train_df['D+5_label']
    X_test = test_df.drop(columns=['文章ID', 'post_time', '標記', 'D+5_label', 'month', 'date'], errors='ignore')
    y_test = test_df['D+5_label']

    model = LogisticRegression(class_weight='balanced', max_iter=10000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    temp_result = pd.DataFrame({
        '預測日': test_day,
        '文章時間': test_df['post_time'].values,
        '預測D+5方向': pred,
        '實際D+5方向': y_test.values,
        '命中': (pred == y_test.values).astype(int)
    })

    results.append(temp_result)

# 統計結果
df_result = pd.concat(results).reset_index(drop=True)
accuracy = df_result['命中'].mean() if len(df_result) > 0 else 0

print("\n✅ 每篇文章預測 D+5（每日出手）")

print(f"準確率：{accuracy:.2%}")

# 混淆矩陣
if len(df_result) > 0:
    y_true = df_result['實際D+5方向']
    y_pred = df_result['預測D+5方向']
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    print("\n 混淆矩陣（每篇文章）")
    print("           預測為漲   預測為跌")
    print(f"實際為漲     {cm[0][0]}        {cm[0][1]}")
    print(f"實際為跌     {cm[1][0]}        {cm[1][1]}")

