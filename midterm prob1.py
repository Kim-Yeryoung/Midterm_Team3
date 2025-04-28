import pandas as pd
import numpy as np
HEAD
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
    print("Categorical columns identified:", cat_cols)
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def create_features(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 60, 120], labels=['young', 'adult', 'senior'])
    return df

def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def preprocessing(df):
    df = handle_missing(df)
    df = remove_outliers(df)
    df = create_features(df)
    df = encode_categoricals(df)
    df = normalize_numerics(df)
    return df

df = preprocessing(df)
df.to_csv('processed_1_adults.csv', index = False)
print(df.info())

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩

def encode_categoricals(df_sub):
    for col in df_sub.columns:
        df_sub[col] = LabelEncoder().fit_transform(df_sub[col])
    return df_sub



df = pd.read_csv("/Users/yunachae/Downloads/1_adults.csv", encoding='cp949')
print(df.info())
print(df.describe())
print(df.head(5))

df['sex'] = df['sex'].str.lower().str.strip()
df['sex'] = df['sex'].replace({
    'male': 'male', 'm': 'male', 'man': 'male',
    'female': 'female', 'f': 'female', 'woman': 'female',
    'trans-female': 'female', 'trans woman': 'female',
    'trans male': 'male', 'trans man': 'male',
    'genderqueer': 'others', 'agender': 'others', 'non-binary': 'others',
    'other': 'others', 'none': 'others'
})
df['sex'] = df['sex'].apply(lambda x: x if x in ['male', 'female'] else 'others')
print(df.head(10))


# Label Encoding할 컬럼 선택
df[['education', 'occupation', 'native.country', 'workclass']] = encode_categoricals(df[['education', 'occupation', 'native.country', 'workclass']])
onehot_cols = ['marital.status', 'relationship', 'race', 'sex', 'income']

df = pd.get_dummies(df, columns=onehot_cols)


0cc9d88 (Merge remote-tracking branch 'origin/main')
