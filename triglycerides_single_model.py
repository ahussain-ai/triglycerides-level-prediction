import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
sample_submission = pd.read_csv('dataset/sample_submission.csv')

# Separate features and target variable
X = train_df.drop(['candidate_id', 'triglyceride_lvl'], axis=1)
y = train_df['triglyceride_lvl']
X_test = test_df.drop(['candidate_id'], axis=1)

# Identify categorical and numerical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

# Bundle preprocessing and modeling code in a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model 
pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = pipeline.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, preds)
score = max(0, 100 - mae)
print('Mean Absolute Error:', mae)
print('Score:', score)

# Predict on the test data
test_preds = pipeline.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({'candidate_id': test_df['candidate_id'], 'triglyceride_lvl': test_preds})

# Ensure the submission file has correct format and size
assert submission.shape == (9600, 2), "Submission file must be 9600 x 2"

# Save submission to CSV
submission.to_csv('submission.csv', index=False)
