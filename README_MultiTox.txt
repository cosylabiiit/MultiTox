Title: MultiTox: Computational Ensemble Stack Model To Predict the Multiclass Toxicity of Proteins

Authors: Harshika Sharma, Mayank Singh Thakur, Avinash Barala, Modh Saad Khan, Ganesh Bagler.

Overview: 
MultiTox is a sequence-based machine learning pipeline for multiclass classification of protein toxins. It integrates ESM-2 embeddings, feature selection, and a stacked ensemble model with multiple base learners (LGBM, KNN, QDA, ExtraTrees, MLP) and an XGBoost meta-classifier.

Highlights:

• MultiTox predicts the multiclass toxicity of proteins based on their mechanism of action in humans.

• The model is trained on a large dataset of 20,029 toxins and 4,245 non-toxins.

• It classifies proteins into neurotoxins, cytotoxins, hemotoxins, and enterotoxins.

• The model is trained by using ESM-2 pre-trained features.

• The model uses a stack ensemble-based approach for the prediction of multiclass toxicity.

Environment & Dependencies: 
Core Dependencies

Python (version ≥3.9)
pandas (version ≥2.0)
numpy (version ≥1.23)
scikit-learn (version ≥1.3)
imbalanced-learn (version ≥0.11)
lightgbm (version ≥4.1)
xgboost (version ≥1.7)
catboost (version ≥1.2)
shap (version ≥0.43)
matplotlib (version ≥3.7)
fair-esm - Latest (ESM-2 embeddings)
torch (version ≥2.0)
tensorflow (version ≥2.15)
keras (version - 2.15.0)
scikeras (version - 0.10.0)

External Tools 
CD-HIT (version 4.8.1) for redundancy reduction.

To install dependencies on Colab:
pip install pandas scikit-learn imbalanced-learn shap matplotlib
pip install fair-esm torch
pip install lightgbm xgboost catboost
pip install tensorflow keras==2.15.0 scikeras==0.10.0

To install CD-HIT on Colab: 
wget https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz
tar -xzvf cd-hit-v4.8.1-2019-0228.tar.gz
cd cd-hit-v4.8.1-2019-0228 && make

Pipeline Overview: 

1. Data Preprocessing: 
	Remove non-natural amino acids (B, J, O, U, X, Z).
	Filter duplicates with CD-HIT (default: 40% identity).
	Clean and format sequences from CSV to FASTA.
2. Feature Extraction:
	Generate ESM-2 embeddings (1280-dim per sequence).
3. Class balancing:
	Apply SMOTE (k=5) for class imbalance only to minor class before train and test splitting.
4. Feature Selection:
	Use Random Forest feature importance.
	Extraction of the top-most important 550 features.
5. Stack Ensemble Training
	Divide the data into train: test in 80:20 split.
Base models:
	LGBMClassifier (n_estimators=50, learning_rate=0.1)
	KNN (n_neighbors=5)
	QDA (default)
	ExtraTreesClassifier (n_estimators=50)
	MLPClassifier (hidden_layer_sizes=(50,), max_iter=200)
Meta model: 
	XGBoost (n_estimators=100, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1).
6. Evaluation: 
	Metrics: Accuracy, Precision, Recall, F1-score, MCC.
7. Model Saving: 
	Base and meta models stored as .pkl files.

Usage Instructions: 
Input Format: CSV file containing column as "Sequences" and "Label" where neurotoxin, cytotoxin, hemotoxin, enterotoxin, and non-toxins are labeled as 1, 2, 3, 4, and 0 respectively.
Outputs: 
Processed embeddings: ESM_embeddings.csv
Balanced dataset: smote_ESM.csv
Trained models:
	meta_classifier.pkl,
	lgbm_model.pkl,
	knn_model.pkl,
	qda_model.pkl,
	et_model.pkl,
	mlp_model.pkl

Web Development: A public web server is available at: https://cosylab.iiitd.edu.in/multitox/ 