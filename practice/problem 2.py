'''
파생함수 (x)
'''
# 실행 상태 초기화로 인해 필요한 패키지 재임포트 및 파일 경로 재설정
import pandas as pd                 # 데이터프레임 조작
import numpy as np                  # 수치 연산
import seaborn as sns               # 시각화
import matplotlib.pyplot as plt     # 시각화
import os                           # 파일/디렉토리 작업


from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩

# 처리할 CSV 파일 리스트
files = [
    "cwurData.csv",
]


# 다양한 인코딩을 시도해 CSV 파일을 읽기 위한 리스트
encodings = ['utf-8', 'cp949', 'euc-kr']
# 앞부터 하나식 시도, 실패 시 다음 인코딩으로 넘어감
# 'utf-8' = 유니코드, 'cp949' = 윈도우 한글, 'euc-kr' = 확장된 유니코드

# CSV 파일을 다양한 인코딩으로 시도하여 읽는 함수
def try_read_csv(filepath):
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, low_memory=False), enc  # 성공 시 데이터프레임 반환
        except Exception: 
            continue
    return None, None  # 인코딩 실패 시 None 반환

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

#데이터프레임을 각 연도별로 나눠 그룹화
def divide_col_year(df):
    df['year'] = df['year'].astype(int, errors='ignore')  # 에러 무시하고 형변환
    print("Checking the data for year values:")
    print(df['year'].unique())

    # unique year 값에 따라 필터링하지 않고 자동으로 분할
    year_dfs = []
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        print(f"Rows for {year}: {len(year_df)}")
        year_dfs.append((year_df, year))

    return year_dfs

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
def normalize_numerics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col != 'year']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# 시각화 함수: 히스토그램, 상관관계 히트맵 저장
def visualize(df, name):
    outdir = f"report/{name}"  # 저장 폴더 생성
    os.makedirs(outdir, exist_ok=True)

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols[:5]:  # 최대 5개 수치형 컬럼
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} 분포')
        plt.savefig(f"{outdir}/{col}_hist.png")
        plt.close()

    # 상관관계 히트맵
    if len(num_cols) >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=False, cmap='coolwarm')
        plt.title("상관관계 히트맵")
        plt.savefig(f"{outdir}/correlation_heatmap.png")
        plt.close()

# 전체 처리 파이프라인 함수: 파일 1개 단위
def process_file(filepath):
    df, enc = try_read_csv(filepath)
    if df is None:
        return f"❌ {filepath} 불러오기 실패"

    filename = os.path.basename(filepath).split('.')[0]  # 파일명 추출
    df = handle_missing(df)
    print(f"결측값 처리 후: {df.shape}")
    df = remove_outliers(df)
    print(f"이상치 제거 후: {df.shape}")
    df = encode_categoricals(df)
    print(f"범주형 인코딩 후: {df.shape}")
    df = normalize_numerics(df)
    print(f"정규화화 후: {df.shape}")
    dfs_by_year = divide_col_year(df)

    for year_df, year in dfs_by_year:
        year_name = f"{filename}_{year}"
        visualize(year_df, year_name)

        output_path = f"processed/{year_name}_cleaned.csv"
        os.makedirs("processed", exist_ok=True)
        year_df.to_csv(output_path, index=False)

    return_msg =  f"✅ {filename}: 처리 완료, 저장 위치 → {output_path}"

    return return_msg


# 파일별 전처리 실행
for path in files:
    process_file(path)

# 처리 결과 리스트 출력
results = [process_file(file) for file in files]
print(results)


