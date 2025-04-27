import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩
import os

df = pd.read_csv("Titanic - Machine Learning from Disaster.csv")
print(df.info())

# 결측값 처리 함수: 수치형 → 중앙값, 범주형 → 최빈값
def handle_missing(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:  # 수치형
            df[col] = df[col].fillna(df[col].median())
        else:  # 범주형 또는 문자열
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

def remove_outliers(df):
    df_cleaned = df.copy()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
    return df_cleaned

def encode_categoricals(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [col for col in cat_cols if col != 'Sex']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def encode_categoricals_onehot(df):
    if 'Sex' in df.columns:
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    return df

def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def preprocessing_with_target(df, target_col):
    # 1. 타겟 분리
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 2. X 데이터 전처리
    X = handle_missing(X)
    X = remove_outliers(X)
    X = encode_categoricals(X)
    X = encode_categoricals_onehot(X)
    X = normalize_numerics(X)

    # 3. X와 y 합치기
    df_processed = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return df_processed

df = preprocessing_with_target(df, 'Survived')
df.to_csv('Titanic - Machine Learning from Disaster', index = False)