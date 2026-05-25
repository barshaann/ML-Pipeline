import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    rename_mapping = {
        'width/ dia': 'width_dia',
        'c/s area': 'cs_area',
        'shape factor': 'shape_factor',
        'ultimate load': 'ultimate_load',
        'w/c': 'w_c'
    }
    df.rename(columns=rename_mapping, inplace=True)
    if 'strength' not in df.columns:
        for col in df.columns:
            if 'strength' in col:
                df.rename(columns={col: 'strength'}, inplace=True)
                break
    return df

def main():
    filepath = 'ML DATA SHEET.xlsx'
    print("--- 1. Loading and Cleaning Data ---")
    try:
        df = load_and_clean_data(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return
    print(f"Dataset shape before processing: {df.shape}")
    target_col = 'strength'
    expected_numeric = ['cement', 'sand', 'water', 'nca', 'rca', 'w_c', 
                        'shape_factor', 'density', 'slump', 'width_dia', 
                        'length', 'cs_area', 'ultimate_load', target_col]
    for col in df.columns:
        if col in expected_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("\n--- 2. Missing Values Summary (Before Imputation) ---")
    missing_summary = df.isnull().sum()
    print(missing_summary[missing_summary > 0])
    if df[target_col].isnull().any():
        df = df.dropna(subset=[target_col])
    if 'doi' in df.columns:
        df = df.drop(columns=['doi'])
        print("Dropped non-predictive metadata column: DOI")
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
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(random_state=42, objective='reg:squarederror')
    if HAS_LGBM:
        models['LightGBM'] = LGBMRegressor(random_state=42, verbose=-1)
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\n--- 3. 5-Fold Cross-Validation on Training Data ---")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'),
                                n_jobs=-1)
        cv_results.append({
            'Model': name,
            'MAE': -scores['test_neg_mean_absolute_error'].mean(),
            'RMSE': -scores['test_neg_root_mean_squared_error'].mean(),
            'R2': scores['test_r2'].mean()
        })
    results_df = pd.DataFrame(cv_results).sort_values(by='RMSE')
    print("\nModel Comparison Table (Sorted by RMSE):")
    print(results_df.to_string(index=False))
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n--- 4. Best Model Selected: {best_model_name} ---")
    best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', models[best_model_name])])
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
    print(f"\nSaved best model pipeline to '{model_filename}'")
    test_results = X_test.copy()
    test_results['actual_strength'] = y_test
    test_results['predicted_strength'] = y_pred
    test_results.to_csv('test_predictions.csv', index=False)
    print("Saved test set predictions to 'test_predictions.csv'")
    model_step = best_pipeline.named_steps['model']
    if hasattr(model_step, 'feature_importances_'):
        print("\n--- 6. Feature Importances ---")
        try:
            cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            if len(categorical_features) > 0 and hasattr(cat_encoder, 'get_feature_names_out'):
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            else:
                cat_feature_names = []
        except Exception:
            cat_feature_names = []
        all_feature_names = numeric_features + list(cat_feature_names)
        importances = model_step.feature_importances_
        if len(importances) == len(all_feature_names):
            fi_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            print(fi_df.head(10).to_string(index=False))

        else:
            print("Could not align feature names with importances.")

if __name__ == "__main__":
    main()
