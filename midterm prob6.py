import pandas as pd
import numpy as np
import datetime

input_file="6_shopping.csv"
#csv파일 읽기
df = pd.read_csv(input_file)

# 1. 전처리 함수: 구매 데이터 전처리 후 고객별 집계
def full_preprocess_customer_segmentation(df):
    df = df.copy()

    # 이상치 제거: 구매 수량과 가격이 모두 0보다 큰 경우만 남김
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    # Customer ID가 없는 데이터 제거 (고객 식별 불가 데이터 제거)
    df = df.dropna(subset=['Customer ID'])

    # 날짜 변환: InvoiceDate 컬럼을 문자열 → datetime 타입으로 변환
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # 날짜 파생변수 생성
    df['Month'] = df['InvoiceDate'].dt.month    #구매 월 (1~12월)
    df['Weekday'] = df['InvoiceDate'].dt.weekday   #    # 구매 요일 (0=월요일, 6=일요일)

    # 총 사용 금액 컬럼 생성: 구매수량 × 단가
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # 고객별 집계
    customer_df = df.groupby('Customer ID').agg({
        'InvoiceDate': 'max',       # 최근 구매일
        'Invoice': 'nunique',       # 구매 건수(주문서 수수)
        'Quantity': 'sum',          # 총 구매 수량
        'TotalPrice': 'sum'         # 총 구매 금액
    }).reset_index()
    
    # 집계된 데이터 컬럼명 정리
    customer_df.columns = ['CustomerID', 'LastPurchaseDate', 'PurchaseCount', 'TotalQuantity', 'TotalSpent']

    # Recency 계산: 기준일(2011-12-10) 기준으로 마지막 구매일과의 차이(일수)
    reference_date = datetime.datetime(2011, 12, 10)
    customer_df['Recency'] = (reference_date - customer_df['LastPurchaseDate']).dt.days

    return customer_df

# 2. RFM 세분화 함수: 고객별 RFM 점수 부여
def rfm_segmentation(customer_df):
    customer_df = customer_df.copy()

    # R_score: Recency를 5구간으로 나누어 점수 부여 (5=가장 최근, 1=가장 오래전)
    customer_df['R_score'] = pd.qcut(customer_df['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    
    # F_score: 구매 건수 기준으로 5구간 점수 부여 (5=많이 구매, 1=거의 구매 안 함)    
    customer_df['F_score'] = pd.qcut(customer_df['PurchaseCount'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    
    # M_score: 총 구매 금액 기준으로 5구간 점수 부여 (5=많이 지출, 1=거의 지출 안 함)
    customer_df['M_score'] = pd.qcut(customer_df['TotalSpent'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)

    # RFM 조합 점수: R, F, M 점수를 문자열로 이어붙임 (예: "545")
    customer_df['RFM_Score'] = customer_df['R_score'].astype(str) + customer_df['F_score'].astype(str) + customer_df['M_score'].astype(str)

    return customer_df

# 3. 고객 세분화 함수: RFM 점수 기반으로 VIP/일반/휴면 고객 분류
def assign_customer_segment(customer_df):
    customer_df = customer_df.copy()
    
    # 개별 고객별로 Segment 분류
    def segment(row):
        if row['RFM_Score'] in ['555', '554', '545', '544']:
            return 'VIP'        # 최근에 자주 구매하고 많이 지출한 VIP 고객
        elif row['RFM_Score'] in ['111', '112', '121', '211', '212']:
            return 'Dormant'    # 오래전에 거의 구매 안 한 휴면 고객
        else:
            return 'General'    # 나머지는 일반 고객

    customer_df['Segment'] = customer_df.apply(segment, axis=1)
    return customer_df

# 4. 전체 파이프라인: 전처리 ➔ RFM 점수화 ➔ Segment 분류까지 통합 수행
def some_function(input_file):
    df = pd.read_csv(input_file)
    df_clean = full_preprocess_customer_segmentation(df)
    df_rfm = rfm_segmentation(df_clean)
    df_segmented = assign_customer_segment(df_rfm)
    return df_segmented


  

# 6. 함수 실행하여 전처리 결과 저장
output_file = some_function(input_file) 


# 7. 전처리 완료된 데이터 CSV 파일로 저장
output_file.to_csv('output_file.csv')

# 결과 확인
output_file
