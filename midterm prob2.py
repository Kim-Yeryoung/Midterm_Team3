#5. 마지막: 파이프 라인 함수화 & 전처리 된 csv 파일 추출
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#파일 넣기기
input_file = "C:/Users/kimye/Desktop/2_Card.csv"

def some_function(input_file):

    #1. 파일 확인: 
    df = pd.read_csv(input_file)

    # 결측값 처리 함수: 수치형 → 중앙값, 범주형 → 최빈값
    def handle_missing(df):
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:  # 수치형
                df[col] = df[col].fillna(df[col].median())
            else:  # 범주형 또는 문자열
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        return df



    # 이상치 제거 함수 (IQR 적용)
    def remove_outliers(df):
        df_cleaned = df.copy()
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

        return df_cleaned

    # 범주형 변수 인코딩 함수 (LabelEncoder 사용, 실패 시 해시 기반)
    def encode_categoricals(df):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if col != 'year']
        for col in cat_cols:
            try:
                df[col] = LabelEncoder().fit_transform(df[col])
            except:
                df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)  # 해시값으로 대체
        return df


    # 수치형 데이터 정규화 함수 (StandardScaler 사용)
    def standard_numerics(df):
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        num_cols = [col for col in num_cols if col != 'year']
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df

    # 수치형 데이터 표준화화 함수 (MinMaxScaler 사용)
    def normalize_numerics(df):
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df

    # 변수 분류 (OneHot & gender or Label)
    label_encode_cols = []
    onehot_cols = []

    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Gender', 'gender', 'Sex', 'sex']:
            n_unique = df[col].nunique()
            if n_unique > 2:
                label_encode_cols.append(col)
            elif n_unique == 2:
                onehot_cols.append(col)
        


 
    return df



#1. 데이터 확인
print(df.info())
print(df.isnull().sum())
#2-1 각 열별 결측 비율 계산
null_ratio = (df.isnull().sum() / len(df)) * 100

# 50% 이상 결측인 열만 골라서 삭제
df = df.drop(columns = null_ratio[null_ratio > 50].index)

#2-2. 중복 제거
df.duplicated()
df = df.drop_duplicates(subset=['업체명', '주업종', '사업자등록번호'])

df.columns
df = df.drop(columns=['Transaction Date'])

#2-3. 결측치 제거(채우기기): 
df=handle_missing(df)

#2-4. outlier 제거(수치화):
df=remove_outliers(df)




#3. 엔코딩(변수 분류): #성별 주의!
onehot_cols, label_encode_cols

df[label_encode_cols]=encode_categoricals(df[label_encode_cols]) # Label
df = pd.get_dummies(df, columns=onehot_cols + ['gender']) #OneHot



#5. 마지막: 파이프 라인 함수화 & 전처리 된 csv 파일 추출




