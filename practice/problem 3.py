import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('서울시 상권분석서비스(추정매출-상권)_2024년.csv')

#IQR 사용자 정의 함수
def remove_outlier(df, columns):
  for col in columns:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)

    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

#2. 결측치 확인 & 없음

def preprocess_data(df):
  #3-1. encoder
  encoder=LabelEncoder()
  for col in ['상권_구분_코드_명', '상권_코드_명', '서비스_업종_코드_명']:
    df[col] = encoder.fit_transform(df[col])
    df[['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명', '서비스_업종_코드',
       '서비스_업종_코드_명']]


  #3-2. IQR 방식 이상치 제거 & scaler


  columns_basket=[['당월_매출_금액', '당월_매출_건수', '주중_매출_금액', '주말_매출_금액'],
   ['월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액', '목요일_매출_금액', '금요일_매출_금액',
       '토요일_매출_금액', '일요일_매출_금액'],
    ['시간대_00~06_매출_금액', '시간대_06~11_매출_금액',
       '시간대_11~14_매출_금액', '시간대_14~17_매출_금액', '시간대_17~21_매출_금액',
       '시간대_21~24_매출_금액'],
    ['남성_매출_금액', '여성_매출_금액'],
     ['연령대_10_매출_금액',
       '연령대_20_매출_금액', '연령대_30_매출_금액', '연령대_40_매출_금액', '연령대_50_매출_금액',
       '연령대_60_이상_매출_금액']]
  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 데이터 불러오기
df = pd.read_csv('서울시 상권분석서비스(추정매출-상권)_2024년.csv')

# 1. 이상치 제거 함수 (단일 컬럼)
def remove_outlier(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# 2. 전처리 함수
def preprocess_data(df):
    # (1) 인코딩
    encoder = LabelEncoder()
    for col in ['상권_구분_코드_명', '상권_코드_명', '서비스_업종_코드_명']:
        df[col] = encoder.fit_transform(df[col])

    # (2) IQR 이상치 제거 + 스케일링
    columns_basket = [
        ['당월_매출_금액', '당월_매출_건수', '주중_매출_금액', '주말_매출_금액'],
        ['월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액', '목요일_매출_금액', '금요일_매출_금액', '토요일_매출_금액', '일요일_매출_금액'],
        ['시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액', '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액'],
        ['남성_매출_금액', '여성_매출_금액'],
        ['연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액', '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액']
    ]
    for columns in columns_basket:
      # 컬럼별 이상치 제거
      for col in columns:
        df = remove_outlier(df, col)
      # 컬럼 그룹 스케일링
      scaler = StandardScaler()
      df[columns] = scaler.fit_transform(df[columns])

    return df


# 3. 실행
df = preprocess_data(df)
print(df.head())





print(preprocess_data(df))