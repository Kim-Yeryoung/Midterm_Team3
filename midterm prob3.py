import pandas as pd
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
    #열 관계 활용
    df['name'] = df['name'].fillna(df.groupby('id')['name'].transform('first'))
    df['host_name'] = df['host_name'].fillna(df.groupby('id')['host_name'].transform('first'))

    df['neighbourhood_group', 'neighbourhood'] = pd.get_dummies(df['neighbourhood_group', 'neighbourhood'])
    df['room_type'] = df['room_type'].map({'Shared room': 1, 'Private room': 2, 'Entire home/apt': 3})



    df['profitability'] = df['price'] * df['minimum_nights'] * df['room_type']
    


    return 'result3.csv'

input_file = "3_AB.csv"
output_file = prepro3(input_file)

