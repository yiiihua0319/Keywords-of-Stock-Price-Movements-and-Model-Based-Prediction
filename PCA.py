import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字
plt.rcParams['font.family'] = 'Heiti TC'   
plt.rcParams['axes.unicode_minus'] = False



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


# PCA 累積解釋變異
pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)


n_components_80 = np.argmax(cumsum >= 0.80) + 1
n_components_95 = np.argmax(cumsum >= 0.95) + 1


plt.figure(figsize=(10, 6))
plt.plot(cumsum, marker='o', color='black')
plt.axhline(y=0.95, color='orange', linestyle='--', label='95% 解釋變異')
plt.axhline(y=0.80, color='blue', linestyle='--', label='80% 解釋變異')
plt.axvline(x=n_components_95, color='orange', linestyle='--')
plt.axvline(x=n_components_80, color='blue', linestyle='--')

plt.text(n_components_80 + 2, 0.82, f"前 {n_components_80} 個主成分 ≈ 80%", color='blue')
plt.text(n_components_95 + 2, 0.96, f"前 {n_components_95} 個主成分 ≈ 95%", color='orange')

plt.title("PCA 累積解釋變異圖", fontsize=16)
plt.xlabel("主成分編號（按解釋力排序）", fontsize=16)
plt.ylabel("累積解釋變異", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

print(f"📐 原始特徵維度數：{X.shape[1]}")
print(f"📉 保留 80% 變異需 {n_components_80} 個主成分")
print(f"📉 保留 95% 變異需 {n_components_95} 個主成分")


pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)