# 下載
!pip install transformers scikit-learn pandas matplotlib seaborn --quiet

import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

df = pd.read_csv("長榮航_新聞_V1.0_3%.csv")  # ← 改成自己的檔名
df = df[['title', '標記']].dropna()
df = df[df['標記'].isin([-1, 1])]
df['label'] = df['標記'].map({-1: 0, 1: 1})  # label 必須是 0 或 1


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['title'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 載入 BERT tokenizer 和 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")
bert.to(device)
bert.eval()

# 特徵提取函式
def get_cls_embeddings(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    return cls_embeddings.detach().cpu().numpy()


# 提取 BERT 特徵
X_train = get_cls_embeddings(train_texts)
X_test = get_cls_embeddings(test_texts)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

import numpy as np


np.save("X_train_bert.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test_bert.npy", X_test)
np.save("y_test.npy", y_test)


svc = LinearSVC(class_weight='balanced')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)


print("✅ 準確率 (Accuracy):", accuracy_score(y_test, y_pred))
print("\n🧾 分類報告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=["跌 (-1)", "漲 (1)"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["預測跌", "預測漲"], yticklabels=["實際跌", "實際漲"])
plt.title("Confusion Matrix (BERT + LinearSVC)")
plt.xlabel("預測類別")
plt.ylabel("實際類別")
plt.tight_layout()
plt.show()
