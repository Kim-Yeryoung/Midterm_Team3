#4. 파이프라인

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
df = pd.read_csv('CARD_SUBWAY_MONTH_202501.csv')
new=df.to_numpy()
new

df=pd.DataFrame(new, columns=['노선명', '역명', '승차총승객수', '하차총승객수', '등록일자',0])
df.drop(columns=[0], inplace=True)

def preprocessor(df):
  #3-1. 정규화
  scaler = StandardScaler()
  df[['승차총승객수', '하차총승객수']] = scaler.fit_transform(df[['승차총승객수','하차총승객수']])

  #3-2. 파생변수
  df['승하차비율'] = df['승차총승객수'] / (df['하차총승객수'] + 1e-6)
  #3-3. encoder
  df['노선명'].value_counts()
  encoder = LabelEncoder()
  df['노선명'] = encoder.fit_transform(df['노선명'])
  df['역명'] = encoder.fit_transform(df['역명'])

  return df



df=preprocessor(df)
df

df.to_csv('processed_df.csv')