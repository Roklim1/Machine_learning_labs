# Lab 04 

## 1. Goal
The goal of this lab was to build a reproducible, leakage-free data engineering pipeline for mixed numerical and categorical data using the UCI Adult dataset. The problem is framed as binary classification to predict individual income exceeding $50K based on 14 input features. Key objectives included executing automated schema, range, missing-value, and duplicate checks; performing a duplicate-conflict label audit; engineering derived features without test set contamination; evaluating baseline Logistic Regression performance with and without engineered features; and exporting a serialized transformer, processed dataset SHA-256 hash, and configuration YAML.

## 2. Method
In this lab, Python was executed inside a virtual environment using Pandas, NumPy, Scikit-learn, and Joblib. The raw dataset was ingested directly via 'fetch_openml(name='adult', version=2, as_frame=True)' without in-place modification and validated against expected dimensions. An automated validation step checked data types, defined ranges, and identified '?' values without dropping rows. Label quality was audited by identifying identical feature rows that mapped to contradictory income labels, and 50 random rows were saved for manual validation.

To ensure zero data leakage, a 70/30 stratified train/test split was performed prior to any data cleaning. Exact duplicates were removed exclusively from the training split while leaving the test split intact. Imputers and outlier clipping boundaries were computed solely on the training partition. Two domain features ('capital_net = capital-gain - capital-loss' and 'hours_per_age = hours-per-week / (age - 16)') were constructed. Continuous variables were standardized with 'StandardScaler', and categorical variables were encoded via 'OneHotEncoder(handle_unknown='ignore')'. Baseline performance was evaluated via 'LogisticRegression' before and after feature addition. Finally, the preprocessor state was saved via 'joblib.dump' to 'results/preprocessor.joblib', and the processed training dataset was hashed using SHA-256 and stored in 'results/config.yaml'.

## 3. Results table or plot

### Ingestion & Validation Checks
- Raw Dimensions: 48,842 rows and 15 columns
- Validation Violations: 0 
- Missing Value Counts:
  - 'workclass': 2,799
  - 'occupation': 2,809
  - 'native-country': 857

### Label-Quality & Integrity Audit
- Duplicate-Conflict Count: 5 feature rows with conflicting income labels
- Manual Verification Sample: 50 rows exported to 'results/audit_sample_50.csv'
- Processed Dataset SHA-256: b4f34ab25817beb9eaedf069cf5fa1ffea5348183c06d984899409683078c572

### Baseline Model Performance (Logistic Regression)
| Pipeline Configuration | Accuracy | F1-Score |
| :--- | :--- | :--- |
| Without Derived Features | 0.8540 | 0.6601 |
| With Derived Features ('capital_net', 'hours_per_age') | 0.8578 | 0.6696 |

### Artifact Verification
- results/preprocessor.joblib
- results/processed_train.csv
- results/audit_sample_50.csv
- results/config.yaml
- results/validation_report.txt
- data_card.md

## 4. Three conclusions
1. Introducing 'capital_net' and 'hours_per_age' increased accuracy from 0.8540 to 0.8578 and boosted the positive-class F1-score from 0.6601 to 0.6696, confirming that non-linear interaction terms provide actionable signal to a linear classifier.
2. Restricting deduplication, percentile clipping, statistical imputation, and scaling parameters strictly to the 70% training split guarantees an unbiased evaluation on the untouched 30% test set.
3. The detection of 5 identical feature vectors assigned to opposing target values demonstrates irreducible labeling noise and latent unobserved variables inherent to the census survey.