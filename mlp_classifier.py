import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import ast

# Define model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load data
df = pd.read_csv('proper_df (7).csv')

# Convert each stringified list into an actual list
df['model_family_vector'] = df['model_family_vector'].apply(ast.literal_eval)

# Convert to numpy array with shape (N, 16)
y = np.stack(df['model_family_vector'].values).astype(np.float32)

# Load BERT embeddings
X = np.load('bert_embeddings (1).npy')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=32, shuffle=True
)

# Train model
model = MLPClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(70):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).numpy()
    y_pred_bin = (y_pred_prob >= 0.2).astype(int)

def exact_match_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

ema = exact_match_accuracy(y_test, y_pred_bin)


print(" Evaluation:")
print("Exact Match Accuracy:", ema)
print("Micro F1 Score:", f1_score(y_test, y_pred_bin, average='micro'))
print("Macro F1 Score:", f1_score(y_test, y_pred_bin, average='macro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred_bin))