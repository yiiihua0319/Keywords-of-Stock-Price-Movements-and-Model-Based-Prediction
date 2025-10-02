df_vec = pd.concat([
    df_labeled[['post_time', '標記']].reset_index(drop=True),
    count_df.reset_index(drop=True)
], axis=1)
df_vec.to_csv("文章關鍵字向量.csv", index=False, encoding="utf-8-sig")