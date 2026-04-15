import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
X_tfidf = np.load("misc/clinicalbert_x.npy")
y = np.load("misc/y.npy", allow_pickle=True)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.15, stratify=y
)

# Train and test
model_bert = LogisticRegression(max_iter=1000)
model_bert.fit(X_train, y_train)
y_bert = model_bert.predict(X_test)

# Get metrics and print it
acc_bert = accuracy_score(y_test, y_bert)
report_bert = classification_report(y_test, y_bert)
print("ClinicalBERT:")
print("Accuracy:", acc_bert)
print(report_bert)

# Load TF-IDF features + labels
X_tfidf = sparse.load_npz("misc/tfidf_x.npz")
y = np.load("misc/y.npy", allow_pickle=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.15, random_state=42, stratify=y
)

# Train and test
model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train, y_train)
y_tfidf = model_tfidf.predict(X_test)

# Get metrics and print it
acc_tfidf = accuracy_score(y_test, y_tfidf)
report_tfidf = classification_report(y_test, y_tfidf)
print("TF-IDF:")
print("Accuracy:", acc_tfidf)
print(report_tfidf)

# Get and visualize word weights
corr = np.abs(model_tfidf.coef_).mean(axis=0)
with open("misc/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
feature_names = vectorizer.get_feature_names_out()

df_corr = pd.DataFrame({
    "word": feature_names,
    "corr": corr
}).sort_values("corr", ascending=False)

print("\nTop predictive words:")
print(df_corr.head(20))

print("\nLeast useful words:")
print(df_corr.tail(20))

# measure variation across classes
variation = np.std(model_tfidf.coef_, axis=0)

df_var = pd.DataFrame({
    "word": feature_names,
    "variation": variation
}).sort_values("variation")

print("\nMost non-specific words:")
print(df_var.head(20))

print("\nMost class-specific words:")
print(df_var.tail(20))


# Export the results
df_corr.to_csv("word_correlation.csv", index=False)
df_var.to_csv("word_variation.csv", index=False)

with open("misc/metrics.txt", "w") as f:
    f.write("ClinicalBERT\n")
    f.write(f"Accuracy: {acc_bert}\n")
    f.write(report_bert)

    f.write("\n\nTF-IDF\n")
    f.write(f"Accuracy: {acc_tfidf}\n")
    f.write(report_tfidf)

# Top and least predictive words
top = df_corr.head(20)
plt.figure()
plt.barh(top["word"], top["corr"])
plt.gca().invert_yaxis()
plt.title("Top Correlated Words")
plt.savefig("top_words.png", bbox_inches="tight")
plt.close()

bottom = df_corr.tail(20)
plt.figure()
plt.barh(bottom["word"], bottom["corr"])
plt.gca().invert_yaxis()
plt.title("Least Correlated Words")
plt.savefig("least_words.png", bbox_inches="tight")
plt.close()

# Specificity of words
spec = df_var.tail(20)
plt.figure()
plt.barh(spec["word"], spec["variation"])
plt.gca().invert_yaxis()
plt.title("Most Class-Specific Words")
plt.savefig("specific_words.png", bbox_inches="tight")
plt.close()

nonspec = df_var.head(20)
plt.figure()
plt.barh(nonspec["word"], nonspec["variation"])
plt.gca().invert_yaxis()
plt.title("Most Non-Specific Words")
plt.savefig("nonspecific_words.png", bbox_inches="tight")
plt.close()

# Histograms
plt.figure()
plt.hist(df_corr["corr"], bins=50)
plt.title("Correlation Distribution")
plt.savefig("correlation_dist.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.hist(df_var["variation"], bins=50)
plt.title("Variation Distribution")
plt.savefig("variation_dist.png", bbox_inches="tight")
plt.close()