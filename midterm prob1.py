import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩


  
input_file="1_adults.csv"
#csv파일 읽기
df = pd.read_csv(input_file, encoding='cp949')
print(df.info())

#결측치 처리: ?로 처리 되어있는 데이터를 NaN값으로 변환
for col in df.columns:
    df[col] = df[col].replace('?', np.nan)

#수치형은 중앙값, 범주형은 최빈값으로 처리
def handle_missing(df):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df

#이상치 제거: IQR을 사용하여 quatile 1 and 3 에서 부터 IQR의 1.5배 이상 넘어가는 값들을 이상치라 판단, 제거
def remove_outliers(df):
    df_cleaned = df.copy()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
    return df_cleaned
#범주형 엔코딩: 성별을 제외한 모든 오브젝트, 카테고리 범주의 데이터 타입 컬럼을 LabelEncoder으로 처리;
def encode_categoricals(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

#성별 컬럼은 onehot encoding사용 ∵컬럼내 데이터의 variation이 적음(Male/Female)
def encode_categoricals_onehot(df):
    if 'sex' in df.columns:
        df = pd.get_dummies(df, columns=['sex'], drop_first=True)
        df.rename(columns={'sex_1': 'sex(Male): 0', 'sex_2': 'sex(Female): 1'}, inplace=True)
        return df
    
#파생변수 생성: 나이를 30마다 구간을 나눠 카테고리화
def create_features(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 60, 120], labels=['young', 'adult', 'senior'])
    return df

#정규화: StandardScaler()를 사용하여 모든 컬럼에 대한 정규화 스케일링 실행
def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df



    

#전처리 파이프라인 실행: 타겟 컬럼을 제외한 모든 컬럼에 대해 전처리 실행
def some_function(input_file, target_col):
    df = pd.read_csv(input_file)
    #1 target column과 그외 컬럼 분리
    y = df[target_col]
    X = df.drop(columns=[target_col])
    #타겟을 제외한 컬럼들에 대하여 전처리 실행
    X = handle_missing(df)
    X = remove_outliers(df)
    X = create_features(df)
    X = encode_categoricals(df)
    X = normalize_numerics(df)

    # 3. X와 y 합치기
    df_processed = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return df_processed

output_file = some_function(input_file, 'income')
df.to_csv('output_file.csv', index = False)
