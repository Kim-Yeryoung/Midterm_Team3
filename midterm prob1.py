import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩


df = pd.read_csv("1_adults.csv", encoding='cp949')
print(df.info())

for col in df.columns:
    df[col] = df[col].replace('?', np.nan)

def handle_missing(df):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
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
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def encode_categoricals_onehot(df):
    if 'sex' in df.columns:
        df = pd.get_dummies(df, columns=['sex'], drop_first=True)
        df.rename(columns={'sex_1': 'sex(Male): 0', 'sex_2': 'sex(Female): 1'}, inplace=True)
        return df

def create_features(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 60, 120], labels=['young', 'adult', 'senior'])
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
    X = handle_missing(df)
    X = remove_outliers(df)
    X = create_features(df)
    X = encode_categoricals(df)
    X = normalize_numerics(df)

    # 3. X와 y 합치기
    df_processed = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return df_processed

df = preprocessing_with_target(df, 'income')
df.to_csv('processed_1_adults.csv', index = False)