import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 사용자 정의 함수 prepro -> 전체 전처리 과정 포괄
def prepro3(input_file):

    # 파일 읽고 df로 저장
    df_raw = pd.read_csv(input_file)
    df = df_raw.copy()

    # 아래 주석 처리한 함수들로 사전에 칼럼별 유니크값, 결측치 비율 파악
    #print(df['longitude'].value_counts())
    #null_ratio = (df.isnull().sum() / len(df)) * 100
    
    # id칼럼 중복치 제거 (기준열이기 때문)
    df['id'] = df['id'].drop_duplicates()
    
    # id열에서 name열에 결측치가 있을 때, id열의 값이 같은 다른 행의 name열 값을 가져다가 결측치 채움
    df['name'] = df['name'].fillna(df.groupby('id')['name'].transform('first'))
    df['host_name'] = df['host_name'].fillna(df.groupby('host_id')['host_name'].transform('first'))

    # neighbourhood_group열 one-hot encoding
    df['neighbourhood_group'] = pd.get_dummies(df['neighbourhood_group'])

    # room_type열 각 등급별 숫자 배당해서 범주형 -> 수치형 데이터 변환
    df['room_type'] = df['room_type'].map({'Shared room': 1, 'Private room': 2, 'Entire home/apt': 3})

    # 파생 변수로 문제에서 구하고자 하는 열 생성 -> 예약가능성, 수익성
    scaler = MinMaxScaler()
    df['reserv_ava'] = scaler.fit_transform(df['availability_365']) # minmax scaler 정규화 #예약 가능한 날짜 수를 백분율처럼 변환함
    df['profitability'] = df['price'] * df['minimum_nights'] * df['room_type'] #수익성은 가격, 최소 숙박일, 객실 등급을 곱해 계산

    # 지역별 예약가능성, 지역별 수익성 계산
    group_profitability = df.groupby('neighbourhood_group')['profitability'].transform('mean')
    group_availability = df.groupby('neighbourhood_group')['reserv_ava'].transform('mean')

    # 지역별 파생 변수(예약가능성, 수익성)로 새로운 열 생성
    df['group_profitability'] = group_profitability
    df['group_reservation_success_rate'] = group_availability

    # 파일 저장, 반환
    df.to_csv('result3.csv')
    return 'result3.csv'

# 최종 파일 input, output
input_file = "3_AB.csv"
output_file = prepro3(input_file)

