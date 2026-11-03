# Data Card

## 1. Source & Provenance
- Dataset Name: UCI Adult Census Income
- Access Point: OpenML repository via fetch_openml with name 'adult' and version 2.
- Origin: Extracted by Barry Becker and Ronny Kohavi from the 1994 US Census Bureau database.

## 2. License & Terms of Use
- License: Public Domain / Creative Commons Attribution 4.0 International.
- Usage: Free to use for academic, educational, and research projects.

## 3. Dataset Size & Structure
- Raw Instances: 48,842 rows and 15 columns, including 14 features and the target label.
- Target Feature: Income category predicting whether annual earnings exceed 50K.
- Numerical Features: Age, final sample weight 'fnlwgt', education years, capital gain, capital loss, and weekly working hours.
- Categorical Features: Workclass, education level, marital status, occupation, relationship, race, sex, and native country.
- Data Splits: 70% stratified training partition with exact duplicates dropped, and an untouched 30% test set of 14,653 rows.

## 4. Known Biases & Skews
- Time Context: The survey reflects the economic realities, wage gaps, and labor market conditions of the United States in 1994.
- Class Imbalance: Significant class imbalance where about 76% of individuals earn 50K or less, and only 24% earn above 50K.
- Demographic Disparities: Evident historical wage gaps across gender and ethnic categories reflecting systemic differences of that era.

## 5. Privacy & Ethics
- Personal Data: Direct identifiers like names, social security numbers, and addresses were never included in the release.
- Sampling Weights: The 'fnlwgt' column represents population weights calculated by the Census Bureau rather than individual tracking data.
- Ethical Guidance: Demographic fields like race and sex should not be used as direct criteria in real-world automated screening to avoid unfair outcomes.