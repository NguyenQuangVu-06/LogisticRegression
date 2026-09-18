# Logistic Regression

## 📘 Introduction
Logistic Regression is a machine learning algorithm used for **binary classification**.
Instead of predicting a continuous value (like Linear Regression), Logistic Regression predicts the **probability** that a sample belongs to class 1 (or 0) using the sigmoid function:
## 🛠️ Key Steps
1. **fit(X, y)**: Train the model on the data.
2. **predict(X)**: Predict the label (0 or 1).
3. **predict_proba(X)**: Predict the probability of belonging to each class.
4. **Model Evaluation**:
- Accuracy
- Precision
- Recall
- F1-score

## 📊 Example using Python & scikit-learn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
# Generate synthetic data
X = np.arange(0,100).reshape(-1,1)
y = np.random.randint(0,2,100)
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# Model coefficients
print("w =", model.coef_)
print("b =", model.intercept_)
