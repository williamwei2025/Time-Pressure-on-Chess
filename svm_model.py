import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('New_Data.csv')  # Update the path to where you've saved the CSV file

# Split the data
train_data = data.iloc[:14000]
validation_data = data.iloc[14000:18000]
test_data = data.iloc[18000:]

# Feature and target setup for training, validation, and testing
X_train = train_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_train = train_data['Predicted_Outcome']

X_validation = validation_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_validation = validation_data['Predicted_Outcome']

X_test = test_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_test = test_data['Predicted_Outcome']

# Possible values of C to test
C_values = [0.01, 0.1, 1, 10, 100]
best_score = 0
best_parameter = None

# Manual hyperparameter tuning for C
for C in C_values:
    model = LinearSVC(penalty='l2', dual=False, C=C)
    model.fit(X_train, y_train)
    score = model.score(X_validation, y_validation)
    if score > best_score:
        best_score = score
        best_parameter = C

# Display best parameter and its score
print(f"Best C value: {best_parameter} with validation accuracy: {best_score}")

# Retrain model on the full training data with the best parameter
model = LinearSVC(penalty='l2', dual=False, C=best_parameter)
model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_accuracy}")

# Optionally, output coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)