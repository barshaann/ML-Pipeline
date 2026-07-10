import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    if any('Unnamed' in str(c) for c in df.columns):
        df = pd.read_excel(filepath, header=1)
    else:
        first_row_str = df.iloc[0].astype(str).str.lower()
        if any('strength' in val or 'cement' in val for val in first_row_str.values):
            df.columns = df.iloc[0]
            df = df.drop(index=0).reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    # Remove units in parentheses to make renaming easier
    df.columns = df.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()
    
    rename_mapping = {
        'width/ dia': 'width_dia',
        'c/s area': 'cs_area',
        'shape factor': 'shape_factor',
        'ultimate load': 'ultimate_load',
        'w/c': 'w_c',
        'nca': 'nca',
        'rca': 'rca',
        'density': 'density',
        'slump': 'slump'
    }
    df.rename(columns=rename_mapping, inplace=True)
    if 'strength' not in df.columns:
        for col in df.columns:
            if 'strength' in col:
                df.rename(columns={col: 'strength'}, inplace=True)
                break
    return df

def main():
    import sys
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger("model_training_report.txt")
    
    filepath = 'ML DATA SHEET.xlsx'
    print("--- 1. Loading and Cleaning Data ---")
    try:
        df = load_and_clean_data(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return
    print(f"Dataset shape before processing: {df.shape}")
    target_col = 'ultimate_load'
    expected_numeric = ['cement', 'sand', 'water', 'nca', 'rca', 'w_c', 
                        'shape_factor', 'density', 'slump', 'width_dia', 
                        'length', 'cs_area', target_col]
    for col in df.columns:
        if col in expected_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'doi' in df.columns:
        df = df.drop(columns=['doi'])
        print("Dropped non-predictive metadata column: DOI")

    print("\n--- 2. Missing Values Summary (Before Imputation) ---")
    missing_summary = df.isnull().sum()
    print(missing_summary[missing_summary > 0])
    if df[target_col].isnull().any():
        df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"\nChosen features: {list(X.columns)}")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- 3. Advanced Pipeline: Tuning & Log Transformation ---")
    
    # 1. Gradient Boosting (Grid Search)
    print("Tuning Gradient Boosting...")
    gb_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(regressor=GradientBoostingRegressor(random_state=42), func=np.log1p, inverse_func=np.expm1))
    ])
    gb_params = {
        'model__regressor__n_estimators': [50, 100, 150],
        'model__regressor__max_depth': [2, 3]
    }
    gb_grid = GridSearchCV(gb_pipe, gb_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    print(f"  Best GB Parameters: {gb_grid.best_params_}")
    best_gb = gb_grid.best_estimator_.named_steps['model']
    
    # 2. Random Forest (Grid Search)
    print("Tuning Random Forest...")
    rf_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(regressor=RandomForestRegressor(random_state=42), func=np.log1p, inverse_func=np.expm1))
    ])
    rf_params = {
        'model__regressor__n_estimators': [50, 100, 150],
        'model__regressor__max_depth': [3, 5]
    }
    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"  Best RF Parameters: {rf_grid.best_params_}")
    best_rf = rf_grid.best_estimator_.named_steps['model']
    
    # 3. Linear Regression (Log Transformed)
    lr_model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)

    print("\n--- 4. Ensembling via Voting Regressor ---")
    voting_model = VotingRegressor(estimators=[
        ('lr', lr_model),
        ('gb', best_gb),
        ('rf', best_rf)
    ])
    
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', voting_model)
    ])
    
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print("\n--- 5. Test Set Evaluation ---")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2:   {test_r2:.4f}")
    
    model_filename = 'best_model_pipeline.joblib'
    joblib.dump(best_pipeline, model_filename)
    print(f"\nSaved advanced ensembled model pipeline to '{model_filename}'")
    
    test_results = X_test.copy()
    test_results['actual_load'] = y_test
    test_results['predicted_load'] = y_pred
    test_results.to_csv('test_predictions.csv', index=False)
    print("Saved test set predictions to 'test_predictions.csv'")
    print("\n--- Execution Complete. Full report saved to 'model_training_report.txt' ---")

if __name__ == "__main__":
    main()
