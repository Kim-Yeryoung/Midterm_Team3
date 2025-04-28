import pandas as pd
import numpy as np
import re

input_file='4_MED_NS.csv'


def some_function(input_file):
    df=pd.read_csv(input_file)
    df = df.copy()

    # 1. AppointmentID 중복 제거 (예약 건별 데이터가 중복될 경우 하나만 남긴다)
    df = df.drop_duplicates(subset=['AppointmentID'])

    # 2. 날짜형 데이터 변환 (문자열 -> datetime으로 변환)
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    # 3. 수치형 데이터 변환 및 이상치 제거
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]      ## 나이 0세 미만, 100세 초과 제거




    # 4. 이진화 (Binary Encoding)
    # 'No-show' (방문 여부) 이진화: No=0, Yes=1
    if 'No-show' in df.columns:
        df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})  
        
    
    # 'Gender' (성별) 이진화: Female=0, Male=1
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
        
        
    # 5. 지역(Neighbourhood) 전처리
    def preprocess_neighbourhood(df, rare_threshold=100):
        df = df.copy()
        # (1) 특수문자, 이상한 글자 제거 (영어, 숫자, 공백만 남기기)

        def clean_neighbourhood(text):
            if pd.isna(text):
                return text
            return re.sub(r'[^A-Za-z0-9 ]+', '', text)

        df['Neighbourhood'] = df['Neighbourhood'].apply(clean_neighbourhood)
        
        # (2) 데이터 개수가 적은 희귀 지역 rare 처리
        counts = df['Neighbourhood'].value_counts()
        frequent_neighbourhoods = counts[counts >= rare_threshold].index

        # (3) 많이 등장한 주요 지역은 1, 드문 지역은 0으로 이진화
        df['Neighbourhood'] = df['Neighbourhood'].apply(lambda x: 1 if x in frequent_neighbourhoods else 0)

        return df

    # 지역 컬럼 전처리 실행
    df = some_function(input_file) 


    return df
# 6. 함수 실행하여 전처리 결과 저장
output_file = some_function(input_file) 


# 7. 전처리 완료된 데이터 CSV 파일로 저장
output_file.to_csv('output_file.csv')

# 결과 확인
output_file