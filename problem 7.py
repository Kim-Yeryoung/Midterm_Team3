import pandas as pd
import numpy as np
import re

def full_preprocess(df):
    df = df.copy()

    ## 1. AppointmentID 중복 제거
    df = df.drop_duplicates(subset=['AppointmentID'])

    ## 2. 날짜 데이터 변환
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    ## 3. 수치형 데이터 변환 및 이상치 제거
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]  # 나이 0~100 사이만

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
