import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def handle_missing(df):
    if df.dtype in ['float64', 'int64']:  # 수치형
        df = df.fillna(df.median())
    else:  # 범주형 또는 문자열
        df = df.fillna(df.mode().iloc[0])
    return df


def prepro3(input_file):

    df_raw = pd.read_csv(input_file)
    df = df_raw.copy()

    #print(df['longitude'].value_counts())
    #null_ratio = (df.isnull().sum() / len(df)) * 100
    df['id'] = df['id'].drop_duplicates()

    handle_missing(df)


    return 'result3.csv'

input_file = "3_AB.csv"
output_file = prepro3(input_file)
