import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#读取训练集、测试集
df_train = pd.read_csv("real_train.csv")
df_test = pd.read_csv("real_test.csv")

# 核心修改：改为 word 级切分，并过滤英文停用词，加入 1-2 个词的组合特征
vectorizer = TfidfVectorizer(
    analyzer='word', 
    stop_words='english', 
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(df_train['text'])
X_test_vec = vectorizer.transform(df_test['text'])

# 调库：朴素贝叶斯
nb = MultinomialNB().fit(X_train_vec, df_train['label'])
pred_nb = nb.predict(X_test_vec)

# 调库：KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(X_train_vec, df_train['label'])
pred_knn = knn.predict(X_test_vec)

print("=== 传统机器学习 ===")
print(f"朴素贝叶斯 准确率: {accuracy_score(df_test['label'], pred_nb):.4f}, Macro-F1: {f1_score(df_test['label'], pred_nb, average='macro'):.4f}")
print(f"KNN       准确率: {accuracy_score(df_test['label'], pred_knn):.4f}, Macro-F1: {f1_score(df_test['label'], pred_knn, average='macro'):.4f}")

import joblib
import os
os.makedirs("checkpoints", exist_ok=True)

# 把向量化器和朴素贝叶斯模型存下来
joblib.dump(vectorizer, "checkpoints/tfidf_vectorizer.pkl")
joblib.dump(nb, "checkpoints/nb_model.pkl")
print("NB 模型已保存！")