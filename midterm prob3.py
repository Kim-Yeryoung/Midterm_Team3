import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



def prepro3(input_file):

    df_raw = pd.read_csv(input_file)
    df = df_raw.copy()

    #print(df['longitude'].value_counts())
    null_ratio = (df.isnull().sum() / len(df)) * 100
    print(null_ratio)
    df['id'] = df['id'].drop_duplicates()





    return 'result3.csv'

input_file = "3_AB.csv"
output_file = prepro3(input_file)
