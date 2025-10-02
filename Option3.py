import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("長榮航_新聞_V1.0_3%.csv", encoding="utf-8-sig")
df['文本'] = df['title'].astype(str) + "。" + df['content'].astype(str)

df_labeled = df[df['標記'].isin([1, -1])].reset_index(drop=True)

docs_pos = df_labeled[df_labeled['標記'] == 1]['文本'].tolist()
docs_neg = df_labeled[df_labeled['標記'] == -1]['文本'].tolist()
docs_all = docs_pos + docs_neg


vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=50000)
X_all = vectorizer.fit_transform(docs_all)
X_pos = vectorizer.transform(docs_pos)
X_neg = vectorizer.transform(docs_neg)
vocab = vectorizer.get_feature_names_out()


def compute_stats(X, total_docs):
    tf = X.sum(axis=0).A1
    df = (X > 0).sum(axis=0).A1
    idf = np.log((total_docs + 1) / (df + 1)) + 1
    tfidf = tf * idf
    return tf, df, tfidf

tf_all, df_all, tfidf_all = compute_stats(X_all, len(docs_all))
tf_pos, df_pos, tfidf_pos = compute_stats(X_pos, len(docs_all))
tf_neg, df_neg, tfidf_neg = compute_stats(X_neg, len(docs_all))

# 全部DF - 漲DF > 30 
vocab_arr = np.array(vocab)
df_all_arr = np.array(df_all)
df_pos_arr = np.array(df_pos)
df_neg_arr = np.array(df_neg)

common_mask = (df_pos_arr > 0) & (df_neg_arr > 0)
junk_mask = (df_all_arr - df_pos_arr) > 30
to_remove_mask = common_mask & junk_mask

filtered_vocab_idx = np.where(~to_remove_mask)[0]  # 保留的索引



vocab = vocab_arr[filtered_vocab_idx]
tf_pos = tf_pos[filtered_vocab_idx]
df_pos = df_pos[filtered_vocab_idx]
tfidf_pos = tfidf_pos[filtered_vocab_idx]
tf_neg = tf_neg[filtered_vocab_idx]
df_neg = df_neg[filtered_vocab_idx]
tfidf_neg = tfidf_neg[filtered_vocab_idx]
tf_all = tf_all[filtered_vocab_idx]
df_all = df_all[filtered_vocab_idx]
tfidf_all = tfidf_all[filtered_vocab_idx]

# 關鍵字分析
def chi2_signed(a, b, c, d):
    N = a + b + c + d
    numerator = (a * d - b * c)
    sign = np.sign(numerator)
    numerator_sq = numerator ** 2 * N
    denominator = (a + b) * (c + d) * (a + c) * (b + d) + 1e-6
    return sign * (numerator_sq / denominator)


def keyword_analysis(target_tf, target_df, target_label, total_tf, total_df, total_docs, target_docs, tfidf_target):
    tf_chi2 = []
    df_chi2 = []
    pmi_list = []
    lift_list = []

    for i in range(len(vocab)):
        a_df = target_df[i]
        b_df = total_df[i] - a_df
        c_df = target_docs - a_df
        d_df = total_docs - target_docs - b_df
        df_chi2_val = chi2_signed(a_df, b_df, c_df, d_df)
        df_chi2.append(df_chi2_val)

        a_tf = target_tf[i]
        b_tf = total_tf[i] - a_tf
        c_tf = target_docs - a_tf
        d_tf = total_docs - target_docs - b_tf
        tf_chi2_val = chi2_signed(a_tf, b_tf, c_tf, d_tf)
        tf_chi2.append(tf_chi2_val)

        p_term = (a_df + b_df) / total_docs
        p_target = target_docs / total_docs
        p_joint = a_df / total_docs
        pmi = np.log2((p_joint + 1e-6) / (p_term * p_target + 1e-6))
        pmi_list.append(pmi)

        lift = (a_df / target_docs + 1e-6) / ((a_df + b_df) / total_docs + 1e-6)
        lift_list.append(lift)

    df_chi2_arr = np.array(df_chi2)
    tfidf_arr = np.array(tfidf_target)
    pmi_arr = np.array(pmi_list)
    lift_arr = np.array(lift_list)

    # 綜合評分
    score = 0.4 * df_chi2_arr + 0.2 * tfidf_arr + 0.2 * pmi_arr + 0.2 * lift_arr

    result = pd.DataFrame({
        '詞': vocab,
        f'{target_label}_TF': target_tf,
        f'{target_label}_DF': target_df,
        f'{target_label}_TF-IDF': tfidf_target,
        '全部TF': total_tf,
        '全部DF': total_df,
        '全部TF-IDF': tfidf_all,
        'TF卡方值(保留正負號)': tf_chi2,
        'DF卡方值(保留正負號)': df_chi2,
        'MI(用DF)': pmi_list,
        'Lift(用DF)': lift_list,
        '綜合排序分數': score
    })

    return result.sort_values("綜合排序分數", ascending=False)



df_pos_result = keyword_analysis(tf_pos, df_pos, "看漲", tf_all, df_all, len(docs_all), len(docs_pos), tfidf_pos)
df_neg_result = keyword_analysis(tf_neg, df_neg, "看跌", tf_all, df_all, len(docs_all), len(docs_neg), tfidf_neg)


df_pos_result.to_csv("看漲關鍵字.csv", index=False, encoding="utf-8-sig")
df_neg_result.to_csv("看跌關鍵字.csv", index=False, encoding="utf-8-sig")
print(" 已完成，看漲關鍵字.csv 與 看跌關鍵字.csv 已輸出。")


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv("長榮航_新聞_V1.0_3%.csv", encoding="utf-8-sig")
df_labeled = df[df['標記'].isin([1, -1])].reset_index(drop=True)

bull_vocab = pd.read_csv("看漲關鍵字.csv", encoding="utf-8-sig")['詞'].tolist()
bear_vocab = pd.read_csv("看跌關鍵字.csv", encoding="utf-8-sig")['詞'].tolist()


if 'content' in df_labeled.columns:
    all_docs = df_labeled['title'].fillna('') + '。' + df_labeled['content'].fillna('')
else:
    all_docs = df_labeled['title'].fillna('')
all_docs = all_docs.tolist()


y = df_labeled["標記"].values


bull_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), vocabulary=bull_vocab)
bear_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), vocabulary=bear_vocab)

X_bull = bull_vectorizer.fit_transform(all_docs)
X_bear = bear_vectorizer.fit_transform(all_docs)

df_bull = pd.DataFrame(X_bull.toarray(), columns=["漲_" + w for w in bull_vectorizer.get_feature_names_out()])
df_bear = pd.DataFrame(X_bear.toarray(), columns=["跌_" + w for w in bear_vectorizer.get_feature_names_out()])
count_df = pd.concat([df_bull, df_bear], axis=1)


X = count_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# Linear SVC
model = LinearSVC(class_weight='balanced', max_iter=10000)
model.fit(X_train, y_train)

#Random Forest
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)

# Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)

# # # KNN
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5)  # 可調整 k 值
# model.fit(X_train, y_train)

# LogisticRegression
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=50000)
# model.fit(X_train, y_train)

# # Naive Bayes
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()
# model.fit(X_train, y_train)



y_pred = model.predict(X_test)


print("準確率 (Accuracy):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=[-1, 1], target_names=["跌 (-1)", "漲 (1)"]))


cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("混淆矩陣")
plt.xticks([0, 1], ["跌 (-1)", "漲 (1)"])
plt.yticks([0, 1], ["跌 (-1)", "漲 (1)"])
plt.xlabel("預測")
plt.ylabel("實際")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.show()
