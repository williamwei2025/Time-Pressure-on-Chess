import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('New_Data.csv')  # Update the path to where you've saved the CSV file

# Split the data
train_data = data.iloc[:16000]
test_data = data.iloc[16000:]  

# Prepare features and target variable for training
X_train = train_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_train = train_data['Predicted_Outcome']

# Prepare features and target variable for testing
X_test = test_data[['Stockfish_Evaluation', 'Time_Left', 'Pieces_Left', 'Elo_Difference']]
y_test = test_data['Predicted_Outcome']

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the coefficients of the logistic regression equation
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

import statsmodels.api as sm

# Adding constant to the features for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit the model using statsmodels
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Print the summary of the logistic regression model
print(result.summary())


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Precision, Recall, and F1 Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)