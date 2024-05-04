import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and split data
data = pd.read_csv('New_Data.csv')
train_data = data.iloc[:14000]
validation_data = data.iloc[14000:18000]
test_data = data.iloc[18000:]

# Set features and targets for training, validation, and testing
X_train = train_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_train = train_data['Predicted_Outcome']
X_validation = validation_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_validation = validation_data['Predicted_Outcome']
X_test = test_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_test = test_data['Predicted_Outcome']

# Parameters to test
C_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
gamma_values = ['auto', 0.001, 0.01, 0.1, 1]
best_score = 0
best_params = {'C': None, 'gamma': None}

# Testing different combinations of parameters
for C in C_values:
    for gamma in gamma_values:
        svc = SVC(kernel='rbf', C=C, gamma=gamma)
        svc.fit(X_train, y_train)
        score = svc.score(X_validation, y_validation)
        if score > best_score:
            best_score = score
            best_params = {'C': C, 'gamma': gamma}

# Output the best parameters and their score
print("Best parameters:", best_params)
print("Best score on validation set: {:.2f}".format(best_score))

# Retrain with the best parameters on the combined training and validation set
X_combined = pd.concat([X_train, X_validation])
y_combined = pd.concat([y_train, y_validation])
svc_best = SVC(kernel='rbf', **best_params)
svc_best.fit(X_combined, y_combined)

# Evaluate on the test set
y_test_pred = svc_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
