import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 정규화 및 범주형 인코딩


df = pd.read_csv("1_adults.csv", encoding='cp949')
print(df.info())
print(df.describe())
print(df.head(5))