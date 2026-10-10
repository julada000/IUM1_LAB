from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine

#Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "wine.csv"


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