import pandas as pd
from sklearn.preprocessing import LabelEncoder  # ← 이 줄 추가!


# 파일 경로는 현재 작업 중인 폴더 기준으로 작성
df = pd.read_csv("train.csv")  # 파일이 같은 폴더에 있는 경우


# 🔹 3. 사용할 열만 선택
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print("처음 5행 미리보기:")
print(df.head())

# 🔹 실습 1: 결측치 처리

# 🧪 결측치 개수 확인
print("\n결측치 개수:")
print(df.isnull().sum())

# ✅ (1) Age를 전체 평균으로 채우기
df['Age_mean'] = df['Age'].fillna(df['Age'].mean())

# ✅ (2) Age를 성별 그룹 평균으로 채우기
df['Age_group'] = df.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mean()))

# ✅ (3) Embarked를 최빈값(가장 많이 나온 값)으로 채우기
most_common = df['Embarked'].mode()[0]
df['Embarked_fill'] = df['Embarked'].fillna(most_common)

# 🔹 실습 2: 인코딩

# ✅ (1) Label Encoding: Sex 컬럼
le = LabelEncoder()
df['Sex_label'] = le.fit_transform(df['Sex'])  # male → 1, female → 0

# ✅ (2) One-Hot Encoding: Embarked_fill 컬럼
df = pd.get_dummies(df, columns=['Embarked_fill'], prefix='Embarked')

# 🔹 실습 3: GroupBy와 결측치 채우기

# ✅ 성별별 생존률
print("\n성별 생존률:")
print(df.groupby('Sex')['Survived'].mean())

# ✅ Pclass + Sex 조합별 생존률
print("\nPclass + Sex 생존률:")
print(df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack())

# ✅ Pclass + Sex 조합별 평균 나이로 결측치 채우기
df['Age_filled'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))

# 🔹 실습 4: 나이 구간 만들고 생존률 분석

# ✅ 연령대 함수 정의
def age_group(age):
    if pd.isnull(age): return 'Unknown'
    if age <= 12: return 'Child'
    elif age <= 19: return 'Teen'
    elif age <= 35: return 'Young Adult'
    elif age <= 60: return 'Adult'
    else: return 'Senior'

# ✅ Age_filled를 기반으로 연령대 구간 만들기
df['Age_band'] = df['Age_filled'].apply(age_group)

# ✅ 연령대 + 성별 생존률
print("\n연령대 + 성별 생존률:")
print(df.groupby(['Age_band', 'Sex'])['Survived'].mean().unstack())

# 🔚 완료 메시지
print("\n🎉 전처리 실습 완료! 이제 머신러닝 모델에 데이터를 넣을 준비가 되었습니다.")

