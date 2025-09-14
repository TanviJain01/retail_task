import numpy as np
import pandas as pd

def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma

def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def select_top_features_by_correlation(df, target_col, top_k=25):
    correlations = df.corrwith(df[target_col]).abs().sort_values(ascending=False)
    correlations = correlations.dropna()
    top_features = correlations.head(top_k + 1).index.tolist()
    if target_col in top_features:
        top_features.remove(target_col)
    else:
        top_features = top_features[:-1]
    return top_features

def preprocess_data(csv_path, target_col="avg_purchase_value"):
    df = pd.read_csv(csv_path)
    
    id_cols = ['transaction_id', 'product_id', 'promotion_id', 'customer_zip_code', 'store_zip_code']
    date_cols = ['transaction_date', 'last_purchase_date', 'product_manufacture_date', 
                 'product_expiry_date', 'promotion_start_date', 'promotion_end_date']
    
    cols_to_remove = [col for col in id_cols + date_cols if col in df.columns]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
    high_card_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col and df[col].nunique() > 50:
            high_card_cols.append(col)
    
    if high_card_cols:
        df = df.drop(columns=high_card_cols)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        df[col], _ = pd.factorize(df[col])
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        temp_cols = [col for col in numerical_cols if col != target_col]
        temp_df = df[temp_cols]
        temp_df, dropped_cols = remove_highly_correlated_features(temp_df, threshold=0.9)

        remaining_cols = list(temp_df.columns) + [target_col]
        categorical_remaining = [col for col in df.columns if col not in numerical_cols]
        df = df[remaining_cols + categorical_remaining]
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    if len(numeric_features) > 20:
        temp_df = df[numeric_features + [target_col]]
        top_features = select_top_features_by_correlation(temp_df, target_col, top_k=20)
        
        non_numeric_features = [col for col in df.columns if col not in numeric_features and col != target_col]
        final_features = top_features + non_numeric_features
        df = df[final_features + [target_col]]
    
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].values.reshape(-1, 1)
    
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        for col in X.columns:
            if col not in X_numeric.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    X = X.astype(float)
    
    # Standardize features
    Xs, mu, sigma = standardize(X.values)
    
    features = X.columns.tolist()
    
    
    return Xs, y, mu, sigma, features, target_col