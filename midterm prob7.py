import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩

#csv 파일 열기
df = pd.read_csv("7_heart.csv")
print(df.info())
print(df.describe())
print(df.head(5))
#결측치 부재를 df.isnull().sum()으로 확인
print(df.isnull().sum())

#이상치 제거: IQR을 사용하여 quatile 1 and 3 에서 부터 IQR의 1.5배 이상 넘어가는 값들을 이상치라 판단, 제거
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

#파생변수 생성: 나이를 40마다 구간을 나눠 카테고리화
def create_features(df):
    #나이를 40마다 구간을 나눠 카테고리화
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 120], labels=['young', 'middle-aged', 'old'])
    #tresbps(안정시 심박수)를 사용하여 평균 130이 넘을시 고혈압으로 변수 생성
    df['high_blood_pressure'] = (df['trestbps'] > 130).astype(int)
    #col(콜레스테롤 수치)가 240을 넘을시 콜레스테롤 수치가 높음으로 변수 생성
    df['high_cholesterol'] = (df['chol'] > 240).astype(int) 
    #thalach(운동 후 최고 심박수)가 100 보다 작을시 low_max_hr 변수 생성
    df['low_max_hr'] = (df['thalach'] < 100).astype(int) 
    #위 세 항목과 fbs(공복혈당), 그리고 oldpeak(운동 중 ST 저하량)이 threshold(2)를 넘으면 포함하여 총 다섯가지의 분석 데이터를 가지고 새로운 컬럼 위험도(점수)를 생성함
    df['risk_score'] = (
        df['high_blood_pressure'] + 
        df['high_cholesterol'] + 
        df['low_max_hr'] + 
        df['fbs'] + 
        (df['oldpeak'] > 2).astype(int)
    )
    return df

#정규화: StandardScaler()를 사용하여 모든 컬럼에 대한 정규화 스케일링 실행
def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

#전처리 파이프라인 실행. 모든 컬럼에 대해 전처리 실행
def preprocessing(df):
    df = remove_outliers(df)
    df = create_features(df)
    df = normalize_numerics(df)
    return df

df = preprocessing(df)
df.to_csv('processed_7_heart.csv', index = False)
print(df.info())
