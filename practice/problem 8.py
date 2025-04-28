### midtermtest/process 8.py
# 1. 데이터 전처리  
# 2. 결측치 처리
# 3. 날짜 처리
# 4. 국가 처리
# 5. 장르 처리
# 6. 출연진 및 감독 처리
# 7. 상위 출연진 N명 추출 및 원핫 인코딩
# 8. 설명 텍스트 정제


import pandas as pd
import re
from collections import Counter

def preprocess_netflix(df, top_n_cast=20):
    # 1. 날짜 처리
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    # 2. 국가 처리
    df['country'] = df['country'].fillna('Unknown').str.strip()
    df['main_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())

    # 3. 장르 처리 → 리스트화
    df['genres'] = df['listed_in'].fillna('').apply(lambda x: [g.strip() for g in x.split(',')])

    # 4. 장르 원핫 인코딩
    df_exploded = df.explode('genres')
    genre_dummies = pd.get_dummies(df_exploded['genres'])
    df_genres_encoded = genre_dummies.groupby(df_exploded.index).sum()
    df = pd.concat([df, df_genres_encoded], axis=1)



    # 5. 출연진 및 감독 처리
    df['cast'] = df['cast'].fillna('')
    df['director'] = df['director'].fillna('')
    df['cast_list'] = df['cast'].apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df['director_list'] = df['director'].apply(lambda x: [d.strip() for d in x.split(',') if d.strip()])

    # 6. 상위 출연진 N명 추출 및 원핫 인코딩
    cast_counter = Counter()
    df['cast_list'].apply(lambda x: cast_counter.update(x))
    top_cast = set([name for name, count in cast_counter.most_common(top_n_cast)])

    df_cast = df[['show_id', 'cast_list']].explode('cast_list')
    df_cast = df_cast[df_cast['cast_list'].isin(top_cast)]
    cast_dummies = pd.get_dummies(df_cast['cast_list'])
    df_cast_encoded = df_cast.join(cast_dummies).groupby('show_id').sum()

    # show_id 기준 병합
    df = df.merge(df_cast_encoded, on='show_id', how='left').fillna(0)

    # 7. 설명 텍스트 정제
    df['description_clean'] = (
        df['description']
        .fillna('')
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
    )

    return df