## 중복제거 및 결측치 처리
## 날짜 데이터 변환
## 수치형 데이터 변환
## 이상값 처리 (예: 'ERROR' → NaN)
import pandas as pd
import numpy as np

def preprocess_transactions(df):
    # 1. 이상값 제거 (예: 'ERROR' → NaN)
    df.replace('ERROR', pd.NA, inplace=True)
    df.replace('UNKNOWN', pd.NA, inplace=True)

    # 2. 수치형 변환
    for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 날짜 변환
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

    # 4. 중복 제거
    df = df.drop_duplicates(subset='Transaction ID')

    #특정행 결측치 제거
    #해당 리스트에 컬럼명 작성
    cols_to_check = ['Item', 'Quantity', 'Price Per Unit', 'Total Spent', 'Payment Method', 'Location']
    df_cleaned = df.dropna(subset=cols_to_check)

    return df