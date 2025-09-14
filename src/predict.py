import numpy as np
import pandas as pd
import pickle
import argparse

def mse(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return float(np.mean((y_true - y_pred)**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot)

def standardize_with_params(X, mu, sigma):
    return (X - mu) / sigma

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def select_top_features_by_correlation(df, target_col, top_k=50):
    correlations = df.corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_features = correlations.head(top_k + 1).index.tolist()
    if target_col in top_features:
        top_features.remove(target_col)
    else:
        top_features = top_features[:-1]
    return top_features

def preprocess_inference_data(df, features, target_col):
    id_cols = ['transaction_id', 'product_id', 'promotion_id']
    date_cols = ['transaction_date', 'last_purchase_date', 'product_manufacture_date', 'product_expiry_date', 'promotion_start_date', 'promotion_end_date']
    
    cols_to_remove = []
    for col in id_cols + date_cols:
        if col in df.columns:
            cols_to_remove.append(col)
    
    df = df.drop(columns=cols_to_remove, errors='ignore')
    
    high_cardinality_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col and df[col].nunique() > 100:
            high_cardinality_cols.append(col)
    
    df = df.drop(columns=high_cardinality_cols, errors='ignore')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        df[col], _ = pd.factorize(df[col])
    
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
  
    X = df[available_features]
    
    for missing_feat in missing_features:
        X[missing_feat] = 0
    
    X = X[features]
    
    return X.astype(float)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--metrics_output_path', required=True)
    parser.add_argument('--predictions_output_path', required=True)
    args = parser.parse_args()
    
    with open(args.model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    mu = model_data['mu']
    sigma = model_data['sigma']
    features = model_data['features']
    target_col = model_data['target_col']
    
    df = pd.read_csv(args.data_path)
    
    y_true = df[target_col].values.reshape(-1, 1)
    
    X = preprocess_inference_data(df, features, target_col)
    
    Xs = standardize_with_params(X.values, mu, sigma)
    
    if hasattr(model, 'predict'):
        y_pred = model.predict(Xs)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

    
    mse_val = mse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    
    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse_val:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_val:.2f}\n")
    
    pred_df = pd.DataFrame(y_pred.ravel())
    pred_df.to_csv(args.predictions_output_path, index=False, header=False)
    
if __name__ == "__main__":
    main()