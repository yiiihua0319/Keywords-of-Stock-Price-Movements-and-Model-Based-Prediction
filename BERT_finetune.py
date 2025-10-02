
import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    Trainer, TrainingArguments
)
from transformers.training_args import TrainingArguments
from transformers import Trainer

from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("長榮航_新聞_V1.0_3%.csv")
df = df[['title', 'content', '標記']].dropna()
df['text'] = df['title'].fillna('') + '。' + df['content'].fillna('')
df = df[df['標記'].isin([-1, 1])]
df['label'] = df['標記'].map({-1: 0, 1: 1})


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# 自動計算類別權重
class_weights_list = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights_list, dtype=torch.float).to(device)

# 轉換為 Hugging Face Datasets
train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

# Tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text"]).with_format("torch")
test_dataset = test_dataset.remove_columns(["text"]).with_format("torch")

# Fine-tune BERT 模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",     
    save_strategy="epoch",            
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.03,
    load_best_model_at_end=True,
    save_total_limit=1,
    logging_dir="./logs",
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

os.environ["WANDB_DISABLED"] = "true"  # 關閉 wandb

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 自動找出 valid loss 最低的 checkpoint
def get_best_checkpoint_by_loss(results_dir="./results"):
    best_loss = float("inf")
    best_checkpoint = None
    for entry in os.scandir(results_dir):
        if entry.is_dir() and "checkpoint" in entry.name:
            eval_file = os.path.join(entry.path, "trainer_state.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    for record in state.get("log_history", []):
                        if "eval_loss" in record:
                            loss = record["eval_loss"]
                            if loss < best_loss:
                                best_loss = loss
                                best_checkpoint = entry.path
    return best_checkpoint

best_checkpoint_path = get_best_checkpoint_by_loss("./results")
print(f" Best checkpoint (valid loss lowest): {best_checkpoint_path}")


bert_backbone = BertModel.from_pretrained(best_checkpoint_path).to(device)
bert_backbone.eval()

def get_cls_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_backbone(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 或改 .mean(dim=1)
            all_embeddings.append(cls_embeddings.cpu())
    return torch.cat(all_embeddings, dim=0).numpy()

X_train = get_cls_embeddings(train_texts)
X_test = get_cls_embeddings(test_texts)
y_train = np.array(train_labels)
y_test = np.array(test_labels)


svc = LinearSVC(class_weight="balanced")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)


print("SVC 準確率：", accuracy_score(y_test, y_pred))
print("\n 分類報告：\n", classification_report(y_test, y_pred, target_names=["跌 (-1)", "漲 (1)"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["預測跌", "預測漲"], yticklabels=["實際跌", "實際漲"])
plt.title("Confusion Matrix (BERT fine-tune + SVC)")
plt.xlabel("預測類別")
plt.ylabel("實際類別")
plt.tight_layout()
plt.show()
