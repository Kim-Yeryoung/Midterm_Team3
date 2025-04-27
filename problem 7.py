## 중복제거 및 결측치 처리
## 날짜 데이터 변환
## 수치형 데이터 변환

import pandas as pd
import numpy as np
from scipy import stats


def full_preprocess(df):
    df = df.copy()

    # 1. 이상값 제거 ('ERROR', 'UNKNOWN' → NaN)
    df.replace(['ERROR', 'UNKNOWN'], pd.NA, inplace=True)

    # 2. 수치형 변환
    for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 날짜 변환
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

    # 4. 중복 제거
    df = df.drop_duplicates(subset=['업체명', '주업종', '사업자등록번호'])

    # 5. 특정 컬럼 결측치 제거
    cols_to_check = ['Item', 'Quantity', 'Price Per Unit', 'Total Spent', 'Payment Method', 'Location']
    df = df.dropna(subset=cols_to_check)

    # 6. 이상치 제거 (IQR 방식)
    numeric_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    #6. 이상치 제거 (z-score 방식)
    z_scores = np.abs(stats.zscore(df[numeric_cols]))    
    df = df[(z_scores < 3).all(axis=1)]

    # 7. 이진화
    if 'colunm name' in df.columns:
        df['colunm name'] = df['colunm name'].map({'Y': 1, 'N': 0})

    ## 8. 소재지 분리
    def extract_region_parts(addr):
        if pd.isna(addr) or str(addr).strip() == "":
            return pd.NA, pd.NA, pd.NA

        parts = str(addr).strip().split()
        
        if len(parts) >= 3:
            sido = parts[0]
            sigungu = parts[1]
            rest = ' '.join(parts[2:])
        elif len(parts) == 2:
            sido = parts[0]
            sigungu = parts[1]
            rest = pd.NA
        elif len(parts) == 1:
            sido = parts[0]
            sigungu = pd.NA
            rest = pd.NA
        else:
            sido, sigungu, rest = pd.NA, pd.NA, pd.NA

        return sido, sigungu, rest

    if 'colunm name' in df.columns:
        df[['시도', '시군구', '상세주소']] = df['소재지'].apply(lambda x: pd.Series(extract_region_parts(x)))

    # 9. 원핫인코딩
    if 'Colunm name' in df.columns:
        df = pd.get_dummies(df, columns=['colunm name'], dummy_na=True)


    return df
