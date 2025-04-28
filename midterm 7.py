import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩


df = pd.read_csv("7_heart.csv")
print(df.info())
print(df.describe())
print(df.head(5))
print(df.isnull().sum())


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

def create_features(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 120], labels=['young', 'middle-aged', 'old'])
    df['high_blood_pressure'] = (df['trestbps'] > 130).astype(int)
    df['high_cholesterol'] = (df['chol'] > 240).astype(int) 
    df['low_max_hr'] = (df['thalach'] < 100).astype(int) 
    df['risk_score'] = (
        df['high_blood_pressure'] + 
        df['high_cholesterol'] + 
        df['low_max_hr'] + 
        df['fbs'] + 
        (df['oldpeak'] > 2).astype(int)
    )
    return df


def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def preprocessing(df):
    df = remove_outliers(df)
    df = create_features(df)
    df = normalize_numerics(df)
    return df

df = preprocessing(df)
df.to_csv('processed_7_heart.csv', index = False)
print(df.info())
