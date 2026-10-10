from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "wine.csv"
RESULTS_PATH = BASE_DIR / "wyniki.txt"

#Define experiment configuration
TEST_SIZE = 0.2
RANDOM_SEED = 23

"""Exercise 3: Data Loading and Exploratory Data Analysis"""
#Load the Wine dataset from scikit-learn
wine = load_wine(as_frame=True)

#Create the data directory and save the dataset if it does not exist
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    wine.frame.to_csv(DATA_PATH, index=False)

#Load the data into a DataFrame
df = pd.read_csv(DATA_PATH)

#First 5 rows
print("First 5 rows:")
print(df.head())

#Size of the dataset
print("\nSize of the dataset:")
print(df.shape)

#Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

#Display dataset structure and column data types
print("\nDataset information:")
df.info()

#Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

#Display the distribution of target classes
print("\nTarget class distribution:")
print(df["target"].value_counts().sort_index())

#Check for duplicate records
print("\nNumber of duplicate records:")
print(df.duplicated().sum())


"""Exercise 4: Data Splitting and Initial Model Training"""
#Separate features from the target variable
X = df.drop(columns=["target"])
y = df["target"]

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

#Display training and testing set dimensions
print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

#Initialize the Logistic Regression classifier
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
)

#Train the model using the training dataset
model.fit(X_train, y_train)

print("\nModel training completed successfully!")



"""Exercise 5: Model Evaluation"""
#Generate predictions on the test dataset
y_pred = model.predict(X_test)

#Calculate the model's classification accuracy
accuracy = accuracy_score(y_test, y_pred)

#Generate a detailed classification performance report
report = classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names
)

#Display model evaluation results
print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(report)

#Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Display the confusion matrix
print("\nConfusion Matrix:")
print(cm)



#Save evaluation results to a text file
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print(f"\nEvaluation results saved to {RESULTS_PATH.name}")
