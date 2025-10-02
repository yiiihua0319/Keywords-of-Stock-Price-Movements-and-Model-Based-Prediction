import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字
plt.rcParams['font.family'] = 'Heiti TC'   
plt.rcParams['axes.unicode_minus'] = False  


import pandas as pd
import matplotlib.pyplot as plt

df_news = pd.read_csv("長榮航_新聞_AI.csv", encoding="utf-8-sig")
df_stock = pd.read_csv("長榮航_2618_收盤資料.csv", encoding="utf-8-sig")


df_news["date"] = pd.to_datetime(df_news["post_time"]).dt.date
df_stock["date"] = pd.to_datetime(df_stock["年月日"]).dt.date
df_stock.rename(columns={"收盤價(元)": "收盤價"}, inplace=True)

df_labeled = df_news.groupby("date")["標記"].min().reset_index()
df_merge = pd.merge(df_stock, df_labeled, on="date", how="left")

plt.figure(figsize=(14,6))


plt.bar(df_merge["date"], df_merge["收盤價"], color="lightsteelblue", label="收盤價")

# 紅點
df_down = df_merge[df_merge["標記"] == -1]
plt.scatter(df_down["date"], df_down["收盤價"], color="red", label="利空", zorder=5, s = 5)

# 綠點
df_up = df_merge[df_merge["標記"] == 1]
plt.scatter(df_up["date"], df_up["收盤價"], color="green", label="利多", zorder=5,  s = 5)

# 灰點
df_neutral = df_merge[df_merge["標記"] == 0]
plt.scatter(df_neutral["date"], df_neutral["收盤價"], color="gray", label="無影響", zorder=5,  s = 5)

plt.title("每日收盤價（長條圖）與新聞事件標記", fontsize=16)
plt.xlabel("日期", fontsize=16)
plt.ylabel("收盤價", fontsize=16)
plt.xticks(rotation=45)
plt.legend(fontsize=16)
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
