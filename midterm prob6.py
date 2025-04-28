import pandas as pd
import numpy as np
import re

# 1. 전처리 함수
def full_preprocess_customer_segmentation(df):
    df = df.copy()

    # 이상치 제거
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    # Customer ID 결측치 제거
    df = df.dropna(subset=['Customer ID'])

    # 날짜 변환
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # 날짜 파생변수
    df['Month'] = df['InvoiceDate'].dt.month
    df['Weekday'] = df['InvoiceDate'].dt.weekday

    # 총 사용 금액 컬럼 생성
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # 고객별 집계
    customer_df = df.groupby('Customer ID').agg({
        'InvoiceDate': 'max',       # 최근 구매일
        'Invoice': 'nunique',       # 구매 건수
        'Quantity': 'sum',          # 총 구매 수량
        'TotalPrice': 'sum'         # 총 구매 금액
    }).reset_index()

    customer_df.columns = ['CustomerID', 'LastPurchaseDate', 'PurchaseCount', 'TotalQuantity', 'TotalSpent']

    # Recency 계산
    reference_date = datetime.datetime(2011, 12, 10)
    customer_df['Recency'] = (reference_date - customer_df['LastPurchaseDate']).dt.days

    return customer_df



    ## 4. 이진화 (Label 및 Gender)
    if 'No-show' in df.columns:
        df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})  # 수정: Yes/No로
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

    def preprocess_neighbourhood(df, rare_threshold=100):
        df = df.copy()

        def clean_neighbourhood(text):
            if pd.isna(text):
                return text
            return re.sub(r'[^A-Za-z0-9 ]+', '', text)

        df['Neighbourhood'] = df['Neighbourhood'].apply(clean_neighbourhood)

        counts = df['Neighbourhood'].value_counts()
        frequent_neighbourhoods = counts[counts >= rare_threshold].index

        # 주요 지역이면 1, 나머지 rare 지역이면 0
        df['Neighbourhood'] = df['Neighbourhood'].apply(lambda x: 1 if x in frequent_neighbourhoods else 0)

        return df


    df = preprocess_neighbourhood(df)  # 호출 필요!!!!

    return df
