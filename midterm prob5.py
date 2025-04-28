#5. 마지막: 파이프 라인 함수화 & 전처리 된 csv 파일 추출
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#파일 넣기기
input_file = "C:/Users/kimye/Desktop/5_SOCCER.csv"

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
        

    #2-2. 중복 제거

    # 1. 불필요한 열 제거
    df = df.drop_duplicates()
    df = df.drop(columns=['sofifa_id', 'player_url', 'dob', 'real_face', 'joined', 'contract_valid_until'])
    

    
    # 처리할 컬럼 리스트
    position_cols = [
        'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
        'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm',
        'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb'
    ]

    # +를 기준으로 앞 숫자만 남기기
    for col in position_cols:
        df[col] = df[col].astype(str).str.split('+').str[0]  # +기준 분리해서 앞부분만
        df[col] = pd.to_numeric(df[col], errors='coerce')    # 숫자로 변환
    
    
    #4-2. 표준화화
    inmax_cols = [
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning',
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
        'attacking_short_passing', 'attacking_volleys',
        'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
        'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
        'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
        'mentality_vision', 'mentality_penalties', 'mentality_composure',
        'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',
        'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
        'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm',
        'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb'
    ]
    df[inmax_cols]=normalize_numerics(df[inmax_cols])
    
    
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

    
    #3. 엔코딩(변수 분류): #성별 주의!
    onehot_cols, label_encode_cols

    df[label_encode_cols]=encode_categoricals(df[label_encode_cols]) # Label
    df = pd.get_dummies(df, columns=onehot_cols ) #OneHot

    def age_to_category(age):
        if age <= 21:
            return 'young'
        elif age <= 26:
            return 'prime'
        else:
            return 'veteran'

    df['age_category'] = df['age'].apply(age_to_category)

    # 2. BMI 계산 (kg/m^2)
    df['body_mass_index'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

        # 3. 공격 능력 평균
    df['attacking_ability'] = (df['shooting'] + df['passing'] + df['dribbling']) / 3

    # 4. 수비 능력 평균
    df['defending_ability'] = (df['defending'] + df['physic']) / 2

 
    return df

output_file = some_function(input_file) 



output_file