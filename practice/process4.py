import pandas as pd

def preprocess_disabled_companies(df):
    # 컬럼명 정리 (공백 제거)
    df.columns = df.columns.str.strip()

    # 기업명/업종 중복 제거
    df = df.drop_duplicates(subset=['업체명', '주업종','사업자등록번호' ])


    # 장애인 채용 여부 이진화
    if '장애인 채용 여부' in df.columns:
        df['장애인 채용 여부'] = df['장애인 채용 여부'].map({'Y': 1, 'N': 0})
    
    #특정행 결측치 제거
  
    cols_to_check = ['소재지', '전화번호', '주요 생산품목']
    df = df.dropna(subset=cols_to_check)


    # '소재지' 컬럼 기준 분리
    def extract_region_parts(addr):
        if pd.isna(addr):
            return pd.NA, pd.NA, pd.NA
        parts = addr.split()
        if len(parts) >= 3:
            sido = parts[0]
            sigungu = parts[1]
            rest = ' '.join(parts[2:])
            return sido, sigungu, rest
        else:
            return pd.NA, pd.NA, addr

    df[['시도', '시군구', '상세주소']] = df['소재지'].apply(lambda x: pd.Series(extract_region_parts(x)))

    return df
