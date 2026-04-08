MEDGENAI — Synthetic Healthcare Data Generator
Overview

MEDGENAI is a synthetic healthcare data generation system built using CTGAN and LLM augmentation.
The system generates realistic patient datasets while preserving statistical relationships and protecting patient privacy.

It also allows dynamic generation of additional clinical insights using a large language model.

Repository Structure
Medgenai.ipynb
app.py
diabetes_dataset.csv
ctgan_healthcare.pkl
README.md
Files Description

Medgenai.ipynb

Data preprocessing
Metadata creation
CTGAN model training
Statistical validation
Correlation analysis
Downstream ML evaluation

app.py

Streamlit web application
Loads trained CTGAN model
Generates synthetic healthcare data
Adds LLM-generated columns
Allows CSV download

diabetes_dataset.csv

Real healthcare dataset used for training
Contains demographic, lifestyle, and clinical variables
Features
Synthetic healthcare data generation using CTGAN
Preservation of statistical distributions
Maintenance of feature correlations
HbA1c to diabetes stage consistency enforcement
LLM-based derived column generation
Interactive Streamlit interface
CSV export support
Dataset Description

The dataset includes:

Age
Gender
Ethnicity
Education level
Income level
Employment status
Smoking status
Alcohol consumption
Physical activity
Diet score
Sleep hours
Screen time
Family history diabetes
Hypertension history
Cardiovascular history
BMI
Waist to hip ratio
Systolic BP
Diastolic BP
Heart rate
HbA1c
Diabetes stage
Diagnosed diabetes
Methodology
Step 1: Data Preprocessing
Removed index column
Converted categorical columns
Metadata detection using SDV
Step 2: Model Training

Model used: CTGAN

The model learns:

Feature distributions
Multivariate relationships
Categorical dependencies
Step 3: Synthetic Data Generation

Synthetic patient records generated using:

model.sample(n)
Step 4: Statistical Validation
Distribution comparison
Correlation heatmaps
Clinical rule validation
Step 5: Downstream Utility

RandomForest trained on synthetic data
Tested on real data

Accuracy achieved: ~85%

Streamlit App

The Streamlit app allows:

Selecting number of rows
Choosing columns
Generating synthetic data
Creating derived columns using LLM
Downloading dataset

Run app:

streamlit run app.py
Installation
pip install streamlit pandas sdv google-generativeai numpy
Example Usage

Generate synthetic data:

Select number of rows
Choose columns
Enter prompt
Click "Generate Data"
Download CSV

Example prompt:

Generate risk_level, explanation, lifestyle_advice
Applications
Healthcare research
ML model training without real patient data
Clinical analytics
Data sharing with privacy protection
Medical AI prototyping
Future Improvements
Multi-table healthcare generation
Temporal patient data
Differential privacy
Drug interaction simulation
Real-time API deployment
Conclusion

MEDGENAI demonstrates that synthetic healthcare data can be generated using CTGAN while maintaining statistical properties and clinical relationships. The generated data supports downstream machine learning tasks and enables privacy-preserving healthcare analytics.
