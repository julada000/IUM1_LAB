
from pathlib import Path
import os
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dotenv import load_dotenv


# Exercise 6: Environment-Based Configuration

# Get the path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load configuration settings from the .env file
load_dotenv(BASE_DIR / ".env")

# Set the dataset path, test set size, and random seed
DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data/wine.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "23"))

# Set the path for saving model evaluation results
RESULTS_PATH = BASE_DIR / "wyniki.txt"


# Exercise 3: Data Loading and Exploratory Data Analysis

# Load the Wine dataset from scikit-learn as a DataFrame
wine = load_wine(as_frame=True)

# Create the data directory if it does not already exist
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save the dataset to a CSV file if it has not been saved yet
if not DATA_PATH.exists():
    wine.frame.to_csv(DATA_PATH, index=False)

# Read the dataset from the CSV file
df = pd.read_csv(DATA_PATH)

# Display the first five records
print("First 5 rows:")
print(df.head())

# Display the number of rows and columns
print("\nSize of the dataset:")
print(df.shape)

# Calculate and display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Display column names, data types, and non-null counts
print("\nDataset information:")
df.info()

# Count missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Count samples belonging to each target class
print("\nTarget class distribution:")
print(df["target"].value_counts().sort_index())

# Count duplicate rows in the dataset
print("\nNumber of duplicate records:")
print(df.duplicated().sum())


# Exercise 4: Data Splitting and Initial Model Training

# Separate input features from the target variable
X = df.drop(columns=["target"])
y = df["target"]

# Split the data into training and testing sets while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

# Display the dimensions of the training and testing sets
print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

# Create a pipeline for feature scaling and logistic regression
model = LogisticRegression(random_state=RANDOM_SEED)

# Train the model using the training data
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# Exercise 5: Model Evaluation

# Predict class labels for the test data
y_pred = model.predict(X_test)

# Calculate the classification accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report with precision, recall, and F1-score
report = classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names
)

# Display the accuracy and classification report
print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(report)

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save the accuracy, classification report, and confusion matrix to a text file
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# Confirm that the evaluation results have been saved
print(f"\nEvaluation results saved to {RESULTS_PATH.name}")
