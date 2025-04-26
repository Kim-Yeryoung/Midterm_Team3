import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 모델 평가 함수 (분류)
def evaluate_team_classification(df, label_col='target'):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    y_binned = pd.cut(y, bins=[-float('inf'), -1, 0, 1, float('inf')], labels=["Very Low", "Low", "High", "Very High"])

    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, random_state=42)

    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)

# 연도별 평가 수행
for year in ['2012', '2013', '2014', '2015']:
    print(year)
    processed_file_path = f"C:/Pyth/processed/cwurData_{year}_cleaned.csv"
    df = pd.read_csv(processed_file_path)
    for col in df.columns:
        score = evaluate_team_classification(df, label_col=col)
        if score is not None:
            print(f" Classification {col} Accuracy Score: {score: .4f}")
    print("="*60)


