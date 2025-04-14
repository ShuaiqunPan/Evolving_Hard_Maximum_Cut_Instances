import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import os
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib


# Helper function to preprocess complex numbers in the dataset
def preprocess_complex_numbers(entry):
    """Helper function to convert complex number strings to floats or handle real values."""
    if isinstance(entry, str):  # Check if the entry is a string
        try:
            # Attempt to convert from string to complex and extract the real part
            return complex(entry.strip('()')).real
        except ValueError:
            return np.nan  # Return NaN for invalid conversions
    elif isinstance(entry, (int, float)):
        return entry  # Directly return the entry if it's already a number
    else:
        return np.nan  # Return NaN for any other unsupported types

# Function to load and preprocess data
def load_and_preprocess(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    complex_cols = ['LOG_LARGESTEIGVAL', 'LOG_SECONDLARGESTEIGVAL', 'LOG_SMALLESTEIGVAL']
    for col in complex_cols:
        df[col] = df[col].apply(preprocess_complex_numbers)
    
    columns_to_drop = ['Unnamed: 0', 'GRAPH_IDX', 'NUM_NODES', 'IS_CONNECTED', 'CALCULATED_RATIO', 'LOG_KEMENY_CONSTANT',
                       'MIN_ECC', 'MAX_ECC', 'LOGRATIO_MAXMINECC']

    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Capitalize all column names
    df.columns = [col.upper() for col in df.columns]
    
    return df

# Load datasets with preprocessing
file_path1 = 'global_graph_features_GW_RQAOA_bottom.csv'
file_path2 = 'global_graph_features_GW_RQAOA_top.csv'
df1 = load_and_preprocess(file_path1)
df2 = load_and_preprocess(file_path2)

print(df1.shape)
print(df2.shape)
print("Remaining columns:", df1.columns.tolist())
print("Remaining columns:", df2.columns.tolist())

# Add a target column to distinguish the classes
df1['class'] = 0
df2['class'] = 1

# Combine the datasets
df = pd.concat([df1, df2], ignore_index=True)

# Prepare features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Define the path to save pipelines
pipeline_folder = 'TPOP_20node'
os.makedirs(pipeline_folder, exist_ok=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline_path = os.path.join(pipeline_folder, 'best_pipeline_seed_42.py')

# Initialize TPOT
tpot = TPOTClassifier(
    generations=100,
    population_size=60,
    verbosity=2,
    random_state=42,
    cv=10,
    n_jobs=-1,
    scoring='balanced_accuracy'
)
# Fit TPOT
tpot.fit(X_train, y_train)
tpot.export(pipeline_path)
print("TPOT has completed and exported the best pipeline for seed 42.")

# Save the datasets for future evaluation
dataset_folder = os.path.join(pipeline_folder, 'seed_42')
os.makedirs(dataset_folder, exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(dataset_folder, 'train_test_datasets.pkl'))

print("Training and testing datasets for seed 42 saved successfully.")