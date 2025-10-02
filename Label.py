import pandas as pd

stock_df = pd.read_csv("長榮航_2618_收盤資料.csv")
stock_df['日期'] = pd.to_datetime(stock_df['年月日']).dt.date
stock_df = stock_df[['日期', '收盤價(元)']].sort_values('日期').reset_index(drop=True)

trading_days = stock_df['日期'].tolist()
price_dict = dict(zip(stock_df['日期'], stock_df['收盤價(元)']))

# 找最近三個交易日，並加權投票
def get_weighted_label(news_date):
    next_days = [d for d in trading_days if d > news_date][:3]
    if len(next_days) < 3:
        return 0  

    weights = [0.5, 0.3, 0.2]
    score = 0

    for i in range(3):
        day_prev = next_days[i]
        try:
            day_next = trading_days[trading_days.index(day_prev) + 1]
        except IndexError:
            continue  

        price_prev = price_dict[day_prev]
        price_next = price_dict.get(day_next)
        if price_next is None:
            continue

        change = (price_next - price_prev) / price_prev
        if change > 0.03:
            label = 1
        elif change < -0.03:
            label = -1
        else:
            label = 0

        score += label * weights[i]

    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


news_df = pd.read_csv("長榮航相關_新聞v1.0.csv")
news_df['新聞日'] = pd.to_datetime(news_df['post_time']).dt.date
news_df['標記'] = news_df['新聞日'].apply(get_weighted_label)


news_df.to_csv("長榮航_新聞_V1.0_3%.csv", index=False, encoding='utf-8-sig')
print("✅ 成功！已使用最近三個交易日的漲跌進行加權標記")
