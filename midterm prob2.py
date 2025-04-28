import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder  


df = pd.read_csv("/Users/yunachae/Downloads/2_Card.csv", encoding='cp949')
print(df.info())
print(df.describe())
print(df.head(5))


def remove_outliers(df):
    df_cleaned = df.copy()
    outlier_cols = [
        'LIMIT_BAL', 'AGE', 
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned


def encode_categoricals_onehot(df):

    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'Age_Category']
    for col in cat_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    bins = [18, 30, 50, 100] 
    labels = ['Young', 'Middle-aged', 'Senior']
    df['Age_Category'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    return df

def encode_categoricals_onehot(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df


from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

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
    outlier_cols = [
        'LIMIT_BAL', 'AGE', 
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

def create_features(df):
    bins = [18, 30, 50, 100]
    labels = ['Young', 'Middle-aged', 'Senior']
    df['Age_Category'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    return df

def encode_categoricals_onehot(df):
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'Age_Category']
    for col in cat_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df

def encode_categoricals(df):
    return df  

def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    exclude_cols = ['ID', 'default.payment.next.month']  
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def preprocessing(df):
    df = handle_missing(df)
    df = remove_outliers(df)
    df = create_features(df)
    df = encode_categoricals_onehot(df)
    df = encode_categoricals(df)  
    df = normalize_numerics(df)
    return df


df = preprocessing(df)
df.to_csv('/Users/yunachae/Downloads/2_Card.csv', index=False)
print(df.info())
