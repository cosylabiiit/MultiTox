# -*- coding: utf-8 -*-
"""MultiTox _final_code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/139lYKSb8I1tI66RrKmOrZ60ao0U0uYQc
"""

!pip install pandas scikit-learn shap matplotlib
!pip install fair-esm
!pip install tensorflow
!pip install keras==2.15.0
!pip install scikeras==0.10.0

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

"""# Preprocessing"""

df = pd.read_csv("/content/dataset2_toxin_seq.csv")
print(df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/content/Dataset_2_toxin_detail (1).csv")
df['Sequence'] = df['Sequence'].str.replace('\n', '')
# Remove spaces from the 'Sequence' column
df['Sequence'] = df['Sequence'].str.replace(' ', '')
# Remove rows with NaN values
df.dropna(inplace=True)
df['Sequence'] = df['Sequence'].astype(str)

# Define the set of natural amino acids
natural_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

# Define function to check for non-natural amino acids
def has_non_natural(seq):
    non_natural_amino_acids = {'B', 'O', 'J', 'U', 'X', 'Z'}
    return any(aa in non_natural_amino_acids for aa in seq)

# Count and print the number of rows containing non-natural amino acids
contains_non_natural = df['Sequence'].apply(has_non_natural)
num_non_natural = contains_non_natural.sum()
print("Number of rows containing non-natural amino acids:", num_non_natural)

# Drop rows with non-natural amino acids
rows_to_delete = df[contains_non_natural].index
df.drop(rows_to_delete, inplace=True)


# Display the updated DataFrame
print(df)

"""# CD-HIT Implementation"""

toxin = pd.read_csv("/content/Neuro.csv")

def csv_to_fasta(csv_file, output_file):
    # Read CSV file into a pandas DataFrame
    toxin = pd.read_csv(csv_file)

    # Open the output FASTA file for writing
    with open(output_file, 'w') as fasta_file:
        # Iterate over rows in the DataFrame
        for index, row in toxin.iterrows():
            # Use the index as the unique identifier for the FASTA header
            fasta_file.write(f'>{index}\n')
            # Write the sequence
            fasta_file.write(f'{row["Sequence"]}\n')

# Replace 'input.csv' and 'output.fasta' with your input CSV file and desired output FASTA file paths
csv_to_fasta('/content/Neuro.csv', '/content/Neuro.fasta')

# Commented out IPython magic to ensure Python compatibility.
#Install CD-HIT
!wget https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz
!tar -xzvf cd-hit-v4.8.1-2019-0228.tar.gz
# %cd cd-hit-v4.8.1-2019-0228
!make

# Commented out IPython magic to ensure Python compatibility.
#Run CD-HIT
# Navigate to the CD-HIT directory
# %cd cd-hit-v4.8.1-2019-0228
# Run CD-HIT
!./cd-hit -i /content/Neuro.fasta -o /content/clusters_Neuro0.4.fasta -c 0.4 -n 2

def fasta_to_csv(fasta_file, output_csv):
    sequences = []
    labels = []

    # Open and read the FASTA file
    with open(fasta_file, 'r') as file:
        sequence = ''
        label = ''
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('>'):
                if sequence:  # Save the previous sequence and its label
                    sequences.append(sequence)
                    labels.append(label)
                    sequence = ''  # Reset sequence for the next one
                label = line[1:]  # Get label (omit the '>' character)
            else:
                sequence += line  # Append sequence lines
        # Append the last sequence after the loop ends
        if sequence:
            sequences.append(sequence)
            labels.append(label)

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame({
        'Label': labels,
        'Sequence': sequences
    })
    df.to_csv(output_csv, index=False)

# Replace 'input.fasta' and 'output.csv' with the paths to your FASTA and desired output CSV files
fasta_to_csv('/content/clusters_Neuro0.4.fasta', '/content/cluster_Neuro0.4.csv')

"""# Feature Extraction"""

import pandas as pd
import torch
import esm  # Load the ESM2 model

# Function to load ESM2 model
def load_esm2_model():
    # Load the pre-trained ESM2 model (33M is the smallest, use larger if needed)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval()  # Set the model to evaluation mode
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

# Function to extract features for a single sequence
def extract_features(sequence, model, batch_converter):
    # Create batch with a dummy label for the ESM2 model
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        # Extract the token embeddings
        results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]

        # Average the token representations (excluding the first token ([CLS]) and last token ([EOS]))
        sequence_rep = token_representations[0, 1:-1].mean(0)
    return sequence_rep.cpu().numpy()  # Return as a numpy array

# Function to process the CSV file and extract embeddings
def extract_features_from_csv(csv_file, output_file, sequence_col):
    # Load the CSV file containing protein sequences
    df = pd.read_csv(csv_file)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Print columns to verify the correct sequence column name
    print(f"Columns in CSV: {df.columns}")

    # Load the ESM2 model
    model, batch_converter = load_esm2_model()

    # Initialize an empty list to hold the extracted embeddings
    all_embeddings = []

    # Loop through each protein sequence in the CSV
    for i, sequence in enumerate(df[sequence_col]):
        print(f"Processing sequence {i+1}/{len(df)}")
        embedding = extract_features(sequence, model, batch_converter)
        all_embeddings.append(embedding)

    # Convert the embeddings into a DataFrame
    embedding_df = pd.DataFrame(all_embeddings)

    # Save the embeddings to a new CSV file
    embedding_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

# Main function to run the extraction
if __name__ == "__main__":
    # File paths
    input_csv = "/content/toxin3052.csv"  # Path to your CSV file
    output_csv = "toxin3052_ESM_embeddings.csv"  # Output file for the embeddings

    # The column in the CSV file that contains the protein sequences
    sequence_col = "Sequence"  # Replace with the actual column name in your CSV

    # Extract features and save them to the output CSV
    extract_features_from_csv(input_csv, output_csv, sequence_col)

"""# SMOTE"""

# Load the dataset
df = pd.read_csv('/content/protbert_features.csv')

# Separate features (X) and target variable (y)
X = df.drop(columns=['Label'])  # Adjust 'Label' if your target column has a different name
y = df['Label']

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine resampled features and labels into a single DataFrame
resampled_data = pd.concat(
    [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Label'])], axis=1
)

# Save the resampled data to a CSV file
resampled_data.to_csv('smote_prot.csv', index=False)
print("SMOTE resampled data saved as 'smote_prot.csv'")

"""# Stack Model with Feature Selection"""

# Load dataset
df = pd.read_csv("/content/smote_ESM.csv")

# Handle missing and infinite values
df.fillna(df.median(), inplace=True)
df.replace([float('inf'), float('-inf')], df.max().max(), inplace=True)

# Separate features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Normalize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Selection using RandomForest
selector = SelectFromModel(RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42), threshold='mean')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Define optimized base classifiers
base_classifiers = [
    ('lgbm', LGBMClassifier(n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('qda', QuadraticDiscriminantAnalysis()),  # Added QDA
    ('et', ExtraTreesClassifier(n_estimators=50, random_state=42)),  # Added ExtraTrees
    ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42))  # Added MLP with basic hyperparameters
]

# Meta-feature storage
meta_train = pd.DataFrame()
meta_test = pd.DataFrame()

# Use 3-fold CV instead of 5
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Train base classifiers and get meta-features
for name, model in base_classifiers:
    print(f"Training {name}...")

    # Special handling for SVC (since it's slow with probability=True)
    if name == 'svc':
        model.fit(X_train_selected, y_train)
        svc_proba_model = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)
        svc_proba_model.fit(X_train_selected, y_train)  # Retrain with probability=True for predictions

        train_pred = cross_val_predict(svc_proba_model, X_train_selected, y_train, cv=skf, method='predict_proba', n_jobs=-1)
        test_pred = svc_proba_model.predict_proba(X_test_selected)
    else:
        model.fit(X_train_selected, y_train)
        train_pred = cross_val_predict(model, X_train_selected, y_train, cv=skf, method='predict_proba', n_jobs=-1)
        test_pred = model.predict_proba(X_test_selected)

    # Store predictions as meta-features
    for i in range(train_pred.shape[1]):
        meta_train[f'{name}_class_{i}'] = train_pred[:, i]
        meta_test[f'{name}_class_{i}'] = test_pred[:, i]

# Train meta-classifier (Logistic Regression)
meta_classifier = XGBClassifier(
    n_estimators=100,  # Increase number of trees
    learning_rate=0.05,  # Reduce learning rate
    max_depth=5,  # Limit tree depth
    subsample=0.8,  # Stochastic gradient boosting
    colsample_bytree=0.8,  # Random column subsampling
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=0.1,  # L2 regularization
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)
meta_classifier.fit(meta_train, y_train)

# Predictions
y_train_pred = meta_classifier.predict(meta_train)
y_test_pred = meta_classifier.predict(meta_test)

# Compute evaluation metrics
metrics = lambda y_true, y_pred: {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, average='macro'),
    "Recall": recall_score(y_true, y_pred, average='macro'),
    "F1 Score": f1_score(y_true, y_pred, average='macro'),
    "MCC": matthews_corrcoef(y_true, y_pred)
}

# Print results
print("Training Metrics:", metrics(y_train, y_train_pred))
print("Testing Metrics:", metrics(y_test, y_test_pred))

"""Saving Base and MEta models"""

import pickle

# Save the meta-model
with open('meta_classifier.pkl', 'wb') as f:
    pickle.dump(meta_classifier, f)

# Save the base models
for name, model in base_classifiers:
    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

