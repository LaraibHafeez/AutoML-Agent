import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
filepath = input("Enter your CSV file path: ")
df = pd.read_csv(filepath)
print(df.shape)
print(df.columns.tolist())
print("\n" + "="*60)
print("AUTO ML CHATBOT AGENT")
print("="*60)
print("""
Hello! I am your AutoML Chatbot Agent.
I can:
1. Read your dataset
2. Clean missing values
3. Encode text columns
4. Detect classification or regression problem
5. Train multiple ML models
6. Compare model performance
7. Select the best model
8. Predict new user input
""")
print("\nAvailable Columns:")
for i, col in enumerate(df.columns):
    print(i, "->", col)

target_col = input("\nEnter the target column name you want to predict: ")

if target_col not in df.columns:
    print("Target column not found!")
else:
    data = df.copy()
    data = data.drop_duplicates()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())
    print("Missing values handled successfully.")
    X = data.drop(target_col, axis=1)
    Y = data[target_col]
    if Y.dtype == "object":
        problem_type = "Classification"
    elif Y.nunique() <= 10:
        problem_type = "Classification"
    else:
        problem_type = "Regression"
    print(f"\nChatbot says: This looks like a {problem_type.upper()} problem.")
    target_encoder = None
    if problem_type == "Classification":
        target_encoder = LabelEncoder()
        Y = target_encoder.fit_transform(Y)
        print("\nTarget classes:")
        for i, cls in enumerate(target_encoder.classes_):
            print(i, "=", cls)
    X = pd.get_dummies(X, drop_first=True)
    print("Input features encoded successfully")
    print("Final number of input features:", X.shape[1])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = {}
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42)
        }
        print("\nTraining Classification Models...\n")
        for name, model in models.items():
            if name in ["Logistic Regression", "SVM", "KNN"]:
                model.fit(X_train_scaled, Y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)
            acc = accuracy_score(Y_test, y_pred)
            results[name] = {"model": model, "accuracy": acc}
            print(f"{name} - Accuracy: {round(acc * 100, 2)}%")
            print("-" * 40)
        best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        print("\nChatbot Final Result:")
        print("Best Model:", best_model_name)
        print("Best Accuracy:", round(results[best_model_name]["accuracy"] * 100, 2), "%")

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        print("\nTraining Regression Models...\n")
        for name, model in models.items():
            if name == "Linear Regression":
                model.fit(X_train_scaled, Y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)
            r2 = r2_score(Y_test, y_pred)
            results[name] = {"model": model, "R2": r2}
            print(f"{name} - R2: {round(r2, 2)}")
            print("-" * 40)
        best_model_name = max(results, key=lambda x: results[x]["R2"])
        print("\nChatbot Final Result:")
        print("Best Model:", best_model_name)
        print("Best R2:", round(results[best_model_name]["R2"], 2))
