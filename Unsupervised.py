import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("長榮航_新聞_V1.0_3%.csv", encoding="utf-8-sig")

df["text"] = df["title"].fillna('') + "。" + df["content"].fillna('')


vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
X = vectorizer.fit_transform(df["text"])

# KMeans
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# t-SNE 
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_2d = tsne.fit_transform(X.toarray())

df["cluster"] = cluster_labels
df["tsne_x"] = X_2d[:, 0]
df["tsne_y"] = X_2d[:, 1]

has_label = "label" in df.columns

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="tsne_x", y="tsne_y",
    hue="cluster",  
    palette="Set1",
    style="label" if has_label else None,
    s=60
)
plt.title("t-SNE Visualization of Articles by KMeans Cluster", fontsize=16)
plt.xlabel("t-SNE X")
plt.ylabel("t-SNE Y")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.stats import mode

if has_label:
    true_labels = df["label"].values  
    pred_clusters = df["cluster"].values  

    label_map = {}
    for cluster in range(n_clusters):
        cluster_indices = (pred_clusters == cluster)
        majority_label = mode(true_labels[cluster_indices])[0][0]
        label_map[cluster] = majority_label


    predicted_labels = np.vectorize(label_map.get)(pred_clusters)

    # accuracy
    acc = accuracy_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, pred_clusters)

    print(f" 準確率 Accuracy: {acc:.4f}")
    print(f" 調整蘭德指數 Adjusted Rand Index (ARI): {ari:.4f}")
