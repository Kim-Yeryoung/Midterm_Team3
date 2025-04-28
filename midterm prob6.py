import pandas as pd
import numpy as np
import datetime

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

# 2. RFM 세분화 함수
def rfm_segmentation(customer_df):
    customer_df = customer_df.copy()

    # RFM 점수 부여
    customer_df['R_score'] = pd.qcut(customer_df['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    customer_df['F_score'] = pd.qcut(customer_df['PurchaseCount'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    customer_df['M_score'] = pd.qcut(customer_df['TotalSpent'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)

    # RFM 조합 점수
    customer_df['RFM_Score'] = customer_df['R_score'].astype(str) + customer_df['F_score'].astype(str) + customer_df['M_score'].astype(str)

    return customer_df

# 3. 고객 세분화 함수
def assign_customer_segment(customer_df):
    customer_df = customer_df.copy()

    def segment(row):
        if row['RFM_Score'] in ['555', '554', '545', '544']:
            return 'VIP'
        elif row['RFM_Score'] in ['111', '112', '121', '211', '212']:
            return 'Dormant'
        else:
            return 'General'

    customer_df['Segment'] = customer_df.apply(segment, axis=1)
    return customer_df

# 4. 전체 파이프라인
def full_customer_segmentation_pipeline(df):
    df_clean = full_preprocess_customer_segmentation(df)
    df_rfm = rfm_segmentation(df_clean)
    df_segmented = assign_customer_segment(df_rfm)
    return df_segmented
