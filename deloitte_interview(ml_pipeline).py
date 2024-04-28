import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
import joblib


# Load the dataset
data_path = 'C:\Users\ali_r\Google Drive\job_hunt\deloitte\fraud1.csv'
data = pd.read_csv(data_path)

# Feature Engineering
# Create 'time_of_day' from 'step'
data['time_of_day'] = data['step'] % 24
data.drop(['step'], axis=1, inplace=True)  # Remove 'step' as it's now redundant

# Handling rare categories for 'category' and 'merchant'
threshold = 0.02  # Threshold for rare categories
for column in ['category', 'merchant']:
    frequencies = data[column].value_counts(normalize=True)
    rare_categories = frequencies[frequencies < threshold].index
    data[column] = data[column].replace(rare_categories, 'Other')

# Define categorical and numerical features
categorical_features = ['age', 'gender', 'category', 'merchant']
numerical_features = ['amount', 'time_of_day']  # Now includes 'time_of_day'

# Setting up the preprocessing steps for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Setting up the preprocessing steps for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating a pipeline that combines the preprocessor with a classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Splitting the data
X = data.drop('fraud', axis=1)
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fitting the model
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f'ROC-AUC: {roc_auc}, MCC: {mcc}')

# Assuming 'model' is the pipeline you've created and trained
joblib.dump(model, 'gradient_boosting_model.pkl')