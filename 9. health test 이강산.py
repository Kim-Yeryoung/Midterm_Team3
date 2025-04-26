import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 모델 평가 함수
def evaluate_team_classification(df, label_col='target'):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    y_binned = pd.cut(y, bins=[-float('inf'), -1, 0, 1, float('inf')], labels=["Very Low", "Low", "High", "Very High"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, random_state=42)

    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)

# 전처리된 데이터 불러오기
processed_file_path = "C:/Pyth/processed/국민건강보험공단_건강검진정보_2023_cleaned.csv"
df = pd.read_csv(processed_file_path)

# 모델 평가
for col in df.columns:
    try:
        accuracy = evaluate_team_classification(df, label_col=col)
        print(f" Classification {col} Accuracy Score: {accuracy: .4f}")
    except Exception as e:
        print(f"Error in column {col}: {e}")

print(df.columns.tolist())