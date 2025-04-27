import pandas as pd
import numpy as np
from scipy import stats

# 데이터 불러오기
df = pd.read_csv("StudentsPerformance.csv")  # 파일 이름과 경로는 필요에 따라 수정

# 1. Z-score 이상치 제거
score_columns = ['math score', 'reading score', 'writing score']
z_scores = stats.zscore(df[score_columns])
df_no_outliers = df[(np.abs(z_scores) < 3).all(axis=1)]

print("✅ 이상치 제거 후 데이터 수:", len(df_no_outliers))
print(df_no_outliers[score_columns].describe())

# 2. 범주형 변수 One-Hot 인코딩
categorical_columns = [
    'gender', 'race/ethnicity',
    'parental level of education',
    'lunch', 'test preparation course'
]

df_encoded = pd.get_dummies(df_no_outliers, columns=categorical_columns)

# 3. 결과 출력
print("\n✅ One-Hot 인코딩된 컬럼명:")
print(df_encoded.columns.tolist())

print("\n✅ 전처리된 데이터 일부 미리보기:")
print(df_encoded.head())
