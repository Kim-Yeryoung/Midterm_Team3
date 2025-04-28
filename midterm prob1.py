import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩


df = pd.read_csv("1_adults.csv", encoding='cp949')
print(df.info())


def handle_missing(df):
    for col in df.columns:
        if (df[col] == 0).all() or df[col].isnull().all():
            df.drop(columns=[col], inplace=True)
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
    return df