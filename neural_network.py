import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the data
data = pd.read_csv('data.csv')
X = data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']].values
y = data['Predicted_Outcome'].values  # Assuming 0 for loss, 1 for win/draw

# Splitting and scaling the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=200):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)

# Instantiate the model, loss function, and optimizer
model = BinaryClassifier(input_dim=4)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(30):
    for inputs, labels in train_loader:
        outputs = model(inputs).squeeze()  # Squeeze to remove extra dimensions
        loss = criterion(outputs, labels.float())  # Convert labels to float to match output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Function to evaluate the model with additional metrics
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        predictions = outputs > 0.5
        accuracy = (predictions == y).float().mean()
        # Convert tensors to numpy arrays for scikit-learn
        precision = precision_score(y.numpy(), predictions.numpy())
        recall = recall_score(y.numpy(), predictions.numpy())
        f1 = f1_score(y.numpy(), predictions.numpy())
    return accuracy.item(), precision, recall, f1

# Evaluate the model
train_results = evaluate(model, X_train, y_train)
test_results = evaluate(model, X_test, y_test)
print(f'Train Accuracy: {train_results[0]}, Precision: {train_results[1]}, Recall: {train_results[2]}, F1 Score: {train_results[3]}')
print(f'Test Accuracy: {test_results[0]}, Precision: {test_results[1]}, Recall: {test_results[2]}, F1 Score: {test_results[3]}')
