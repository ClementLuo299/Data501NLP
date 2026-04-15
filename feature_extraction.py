import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from scipy import sparse
import numpy as np
import pickle
import torch

# Use GPU + batch processing
batch_size = 128
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Open the dataset
df = pd.read_csv("data1.csv")

# Parse the words into something usable
def parse_list(s):
    s = s.strip("[]")
    return [w.strip().strip("'").strip('"') for w in s.split(",")]
transcripts = df["parsed_transcript"].apply(parse_list).tolist()
docs = ["".join(t) for t in transcripts]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(docs)

# Save TF-IDF + vectorizer
sparse.save_npz("misc/tfidf_x.npz", X_tfidf)
with open("misc/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# ClinicalBERT
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)
embeddings = []

# Process batches
for i in range(0, len(docs), batch_size):
    # get docs and update progress
    batch = docs[i:i+batch_size]
    print(i)

    # Apply the tokenizer to prep the input and add them to the device
    # We need them to be pytorch tensors
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # We use .no_grad since we are not training, which speeds things up
    # Then we call ClinicalBERT using the inputs
    with torch.no_grad():
      outputs = model(
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"]
    )

    # Move the embeddings out of the GPU if we used it
    cls_embeddings = outputs.last_hidden_state[:,0,:]
    embeddings.append(cls_embeddings.cpu())

# Concatenate all of the batch embeddings into one
X_bert = torch.cat(embeddings)

# Export the result and the labels
np.save("misc/clinicalbert_x.npy", X_bert.cpu().numpy())
y = df["specialty"].to_numpy()
np.save("misc/y.npy", y)


