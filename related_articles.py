import pandas as pd
import re
import os

# 多階段關鍵字設定
company_keywords = ['長榮航', '長榮航空', 'EVA Air', '2618', '張國煒']
impact_keywords = [
    '股價', '上漲', '下跌', 'EPS', '財報', '營收', '配息', '除息',
    '目標價', '罷工', '油價', '匯率', '疫情', '邊境', '解封', '旅遊',
    '免簽', '航班', '載客率', '航線', '戰爭', '台海', '空域'
]

data_sources = {"bda2025_202301-202503_內容數據_新聞.csv": "新聞"}

all_related = []

def is_related(text):
    if pd.isna(text):
        return False
    return any(kw in text for kw in company_keywords) and any(kw in text for kw in impact_keywords)

def clean_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text

for file_name, source in data_sources.items():
    file_path = os.path.join("/Users/candice/Desktop/SeniorYear/BigData/bda2025_mid_dataset", file_name)
    
    df = pd.read_csv(file_path)

    df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['is_related'] = df['combined_text'].apply(is_related)

    df_related = df[df['is_related']].copy()
    df_related['clean_content'] = df_related['content'].apply(clean_text)
    df_related['資料來源'] = source

    all_related.append(df_related)

df_all_related = pd.concat(all_related, ignore_index=True)

total_count = len(df_all_related)
source_count = df_all_related['資料來源'].value_counts()

print(f"總共相關文章數：{total_count} 篇")
print("各來源文章數：")
print(source_count)

output_cols = ['post_time', 'title', 'content', 'clean_content', '資料來源']
df_all_related[output_cols].to_csv("長榮航相關_新聞.csv", index=False, encoding='utf-8-sig')
print("✅ 已輸出檔案：長榮航相關_新聞.csv")