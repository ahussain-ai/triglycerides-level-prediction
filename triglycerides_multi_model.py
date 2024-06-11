import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

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


# Function to remove outliers based on the IQR method
def remove_outliers(df, numerical_cols):
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[numerical_cols] < lower_bound) | (df[numerical_cols] > upper_bound)).any(axis=1)]

# Remove outliers from the training data
X = remove_outliers(X, numerical_cols)
y = y[X.index]  # Keep only the target values corresponding to the remaining data

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

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

# Define the models to evaluate
models = {
    'XGBoost': XGBRegressor(n_estimators=1000, learning_rate=0.08, n_jobs=5),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR()
}

# Dictionary to store the performance of each model
model_performance = {}

# Preprocess and select features within a pipeline
class FeatureSelector:
    def __init__(self):
        self.selected_features = []

    def fit(self, X, y):
        # Correlation analysis
        X_df = pd.DataFrame(X)
        correlation_matrix = X_df.join(pd.DataFrame(y, columns=['triglyceride_lvl'])).corr()
        correlation_with_target = correlation_matrix['triglyceride_lvl'].drop('triglyceride_lvl')
        high_corr_features = correlation_with_target[abs(correlation_with_target) > 0.1].index.tolist()
        
        # Feature importance
        initial_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
        initial_model.fit(X, y)
        feature_importances = initial_model.feature_importances_
        important_features = [X_df.columns[i] for i in range(len(feature_importances)) if feature_importances[i] > 0.01]
        
        # Combine the results
        self.selected_features = list(set(high_corr_features) | set(important_features))

    def transform(self, X):
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

for model_name, model in models.items():
    # Create the pipeline with preprocessing, feature selection, and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', FeatureSelector()),
        ('model', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    preds = pipeline.predict(X_valid)
    
    # Evaluate the model
    mae = mean_absolute_error(y_valid, preds)
    model_performance[model_name] = mae
    print(f'{model_name} - Mean Absolute Error: {mae}')

# Display the performance of each model
print("\nModel Performance:")
for model_name, mae in model_performance.items():
    score = max(0, 100 - mae)
    print(f'{model_name} - Mean Absolute Error: {mae}, Score: {score}')

# Use the best model to predict on the test data
best_model_name = min(model_performance, key=model_performance.get)
best_model = models[best_model_name]

# Hyperparameter tuning with GridSearchCV for the best model
param_grid = {}

if best_model_name == 'XGBoost':
    param_grid = {
        'model__n_estimators': [500, 1000, 1500],
        'model__learning_rate': [0.01, 0.05, 0.9],
        'model__max_depth': [3, 5, 7],
        'max_features': ['auto', 'sqrt']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'model__n_estimators': [100, 200, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7]
    }
elif best_model_name == 'Support Vector Regressor':
    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__epsilon': [0.1, 0.2, 0.5]
    }

# Create the final pipeline with the best model
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', FeatureSelector()),
    ('model', best_model)
])

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(final_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model_tuned = grid_search.best_estimator_

# Preprocessing of validation data, get predictions
preds = best_model_tuned.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, preds)
score = max(0, 100 - mae)
print(f'Tuned {best_model_name} - Mean Absolute Error: {mae}, Score: {score}')

# Preprocess and select features for the test data
X_test_preprocessed = preprocessor.transform(X_test)
X_test_selected = best_model_tuned.named_steps['feature_selection'].transform(X_test_preprocessed)

# Predict on the test data
test_preds = best_model_tuned.named_steps['model'].predict(X_test_selected)

# Create submission dataframe
submission = pd.DataFrame({'candidate_id': test_df['candidate_id'], 'triglyceride_lvl': test_preds})

# Ensure the submission file has correct format and size
assert submission.shape == (9600, 2), "Submission file must be 9600 x 2"

# Save submission to CSV
submission.to_csv('submission.csv', index=False)


