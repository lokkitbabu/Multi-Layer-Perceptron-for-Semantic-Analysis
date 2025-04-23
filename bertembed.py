import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import ast
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

# 📂 Step 3: Load your CSV file (upload manually or mount Drive)

df = pd.read_csv("proper_df.csv")

# 🧹 Step 4: Combine introduction and conclusion
df["combined_text"] = df["abstract_section"] + " " + df["section"]

# 🔧 Step 5: Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
    tokens = [t for t in tokens if t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)

df["clean_text"] = df["combined_text"].apply(preprocess)

# 🎯 Step 6: Labels — convert stringified lists to actual lists
df["labels"] = df["model_family_vector"].apply(ast.literal_eval)

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_mean_pool_embeddings(text_list, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(embeddings)

# print("TF-IDF shape:", X_test.shape)
# # Get feature names
# feature_names = tfidf.get_feature_names_out()

# # Get top features for a few example documents
# for i in range(3):  # First 3 rows
#     row = X[i].toarray()[0]
#     top_indices = row.argsort()[-10:][::-1]  # Top 10 TF-IDF values
#     top_words = [feature_names[j] for j in top_indices]
#     print(f"Doc {i+1} top words:", top_words)
# Load BERT model (MiniLM is fast and solid)
from transformers import AutoTokenizer, AutoModel

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the preprocessed text (or use "combined_text" if you want to compare raw vs. clean)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased').to(device)
model = SentenceTransformer('all-mpnet-base-v2')
bert_embeddings = model.encode(df["clean_text"].tolist(), show_progress_bar=True)
# bert_embeddings = get_mean_pool_embeddings(df["clean_text"].tolist())

X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    bert_embeddings, df["labels"], test_size=0.2, random_state=42
)
print("BERT embeddings shape:", bert_embeddings.shape)
np.save("bert_embeddings.npy", bert_embeddings)
np.save("y_labels.npy", df["labels"].to_numpy())
