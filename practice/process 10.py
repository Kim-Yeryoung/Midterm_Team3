import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_raw = pd.read_csv('/mnt/data/10_survey.csv')

df = df_raw.copy()

# 결측치 처리
df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)
df['work_interfere'].fillna(df['work_interfere'].mode()[0], inplace=True)

# 불필요한 열 제거
df.drop(columns=['state', 'comments', 'Timestamp'], inplace=True)

# Age 이상치 처리
valid_age = df[(df['Age'] > 0) & (df['Age'] < 100)]['Age']
mean_age = int(valid_age.mean())
df.loc[(df['Age'] <= 0) | (df['Age'] > 100), 'Age'] = mean_age

# Gender 재분류
# print(df['Gender'].value_counts())
df['Gender'] = df['Gender'].str.lower().str.strip()
df['Gender'] = df['Gender'].replace({
    'male': 'male', 'm': 'male', 'man': 'male',
    'female': 'female', 'f': 'female', 'woman': 'female',
    'trans-female': 'female', 'trans woman': 'female',
    'trans male': 'male', 'trans man': 'male',
    'genderqueer': 'others', 'agender': 'others', 'non-binary': 'others',
    'other': 'others', 'none': 'others'
})
df['Gender'] = df['Gender'].apply(lambda x: x if x in ['male', 'female'] else 'others')

# 파생 변수
df['is_us'] = df_raw['Country'].apply(lambda x: 1 if x == 'United States' else 0)
df.drop(columns=['Country'], inplace=True)

# 변수 분류
label_encode_cols = []
onehot_cols = []

for col in df.columns:
    if df[col].dtype == 'object' and col != 'Gender':
        n_unique = df[col].nunique()
        if n_unique > 2:
            label_encode_cols.append(col)
        elif n_unique == 2:
            onehot_cols.append(col)

# Label Encoding + MinMax
ordinal_label_maps = {}
for col in label_encode_cols:
    labels = df[col].dropna().unique().tolist()
    label_map = {val: i for i, val in enumerate(sorted(labels))}
    ordinal_label_maps[col] = label_map
    df[col] = df[col].map(label_map)
    df[[col]] = MinMaxScaler().fit_transform(df[[col]])

# One-hot Encoding
df = pd.get_dummies(df, columns=onehot_cols + ['Gender'])

# Age 정규화
df[['Age']] = MinMaxScaler().fit_transform(df[['Age']])

# 저장
df.to_csv('survey_final_custom_encoded.csv', index=False)