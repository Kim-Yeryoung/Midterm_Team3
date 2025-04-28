## 중복제거 및 결측치 처리
## 날짜 데이터 변환
## 수치형 데이터 변환

import pandas as pd
import numpy as np
from scipy import stats
import re


def full_preprocess(df):
    df = df.copy()

    # 1. 이상값 제거 ('ERROR', 'UNKNOWN' → NaN)
    df.replace(['ERROR', 'UNKNOWN'], pd.NA, inplace=True)

    # 2. 수치형 변환
    for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 날짜 변환
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    # 4. 중복 제거
    df = df.drop_duplicates(subset=['업체명', '주업종', '사업자등록번호'])

    # 5. 특정 컬럼 결측치 제거
    cols_to_check = ['Item', 'Quantity', 'Price Per Unit', 'Total Spent', 'Payment Method', 'Location']
    df = df.dropna(subset=cols_to_check)
    # 5. 특정 컬럼 결측치 대체  (평균, 중앙값 등으로 대체)
    df['colmun name'] = df['column name'].fillna(df['colmun name'].median())



    # 7. 이진화
    if 'No-show' in df.columns:
        df['No-show'] = df['No-show'].map({'Y': 1, 'N': 0})
    if 'Gender' in df.columns:
      df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})



    def preprocess_neighbourhood(df, rare_threshold=100):
        df = df.copy()
        
        # 1. 깨진 문자 정리 (영문, 숫자, 공백만 남기기)
        def clean_neighbourhood(text):
            if pd.isna(text):
                return text
            return re.sub(r'[^A-Za-z0-9 ]+', '', text)

        df['Neighbourhood'] = df['Neighbourhood'].apply(clean_neighbourhood)
        
        # 2. 희귀 지역 통합 (rare → 'Other')
        counts = df['Neighbourhood'].value_counts()
        rare = counts[counts < rare_threshold].index
        df['Neighbourhood'] = df['Neighbourhood'].replace(rare, 'Other')

        # 3. One-hot encoding
        df = pd.get_dummies(df, columns=['Neighbourhood'], dummy_na=True)

        return df


    # 9. 원핫인코딩
    if 'Column name' in df.columns:
        df = pd.get_dummies(df, columns=['colunm name'], dummy_na=True)


    return df
