# AutoML Agent 🤖
An intelligent AutoML system that automatically analyzes any CSV dataset,
detects the problem type, trains multiple ML models, and selects the best one.
## What It Does
- Accepts any CSV dataset as input
- Automatically detects Classification or Regression problem
- Handles missing values automatically
- Encodes categorical columns
- Trains multiple ML models simultaneously
- Compares all model results
- Selects and recommends the best model
## Models Used
### Classification
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
## Technologies Used
- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
## How to Run
1. Clone the repository
   git clone https://github.com/LaraibHafeez/AutoML-Agent.git
2. Install dependencies
   pip install pandas numpy scikit-learn matplotlib
3. Run the agent
   python automl_agent.py
4. Enter your CSV file path when prompted
5. Enter the target column name
6. AutoML Agent will do the rest!
## Sample Output:
Training Classification Models...
Logistic Regression  - Accuracy: 70.0%

SVM                  - Accuracy: 50.0%

KNN                  - Accuracy: 60.0%

Decision Tree        - Accuracy: 80.0%

Random Forest        - Accuracy: 70.0%
Best Model: Decision Tree Classifier

Best Accuracy: 80.0%
## Author
Laraib Hafeez - BS Artificial Intelligence Student
