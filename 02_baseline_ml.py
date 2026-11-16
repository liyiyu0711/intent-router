import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#划分训练集和测试集
df = pd.read_csv("data.csv")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 调库：朴素贝叶斯
nb = MultinomialNB().fit(X_train_vec, y_train)
pred_nb = nb.predict(X_test_vec)

# 调库：KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(X_train_vec, y_train)
pred_knn = knn.predict(X_test_vec)

print("=== 传统机器学习 ===")
print(f"朴素贝叶斯 准确率: {accuracy_score(y_test, pred_nb):.4f}, Macro-F1: {f1_score(y_test, pred_nb, average='macro'):.4f}")
print(f"KNN       准确率: {accuracy_score(y_test, pred_knn):.4f}, Macro-F1: {f1_score(y_test, pred_knn, average='macro'):.4f}")