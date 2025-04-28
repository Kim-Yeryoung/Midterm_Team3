# Midterm_Team3
hihi

1. 여기에 각자 맡은 파트 제출해보아요! (미리 제출하는 연습)
2. 주석 다는 법도 알아야할 것 같습니다-->각자 제출한 내용에 대해서 시험 끝나고 주석 달아야 하기 떄문입니다!
3. 과제도 제출하고 주석도 달았으면 slack에 다시 가서 체크 해주세용!


# 1번_김예령
1) 데이터 정보 확인 & 결측치 확인
2) 결측 없었음
3) 다양한 데이터 변환
  3-1) 정규화- 하차, 탑승 수
  3-2) 파생변수 (탑승객/ 하차 승객 비율 변수 만듦)
  3-3) encoder (역명, 노선명)-모두 row가 굉장히 많으므로 One-Hot Encoder 쓰기에는 coloumn이 너무 커진다.
4) 파이프라인

# 2번_이강산
1) 결측값 처리: 수치형 -> 중앙값; 범주형 -> 최빈값
2) 이상치 제거 (IQR)
3) 데이터가 연도별로 나눠저있는 랭킹 -> 연도별로 데이터를 나눌 필요 있음
     3-1) 각 연도별로 그룹화 진행
4) 범주형 처리 -> 안될시 해시값으로 고유 숫자 지정
5) 수치형 처리 -> 모든 값 정규화
6) 시각화(GPT의 '전적으로' 도움을 받음)
7) 각 연도별 전처리, 시각화 파일 저장
   
# 3번_김예령
 1) 이상치 제거 함수 (단일 컬럼)
 2) IQR 이상치 제거 + 스케일링 + IQR 사용자 정의 함수

# 9번_이강산
>>> 숫자만 존재하는 파일이었음
1) 결측값 처리: 수치형 -> 중앙값; 범주형 -> 최빈값
2) 이상치 제거 (IQR)
3) 범주형 처리 -> 안될시 해시값으로 고유 숫자 지정
    3-1)성별이 1과 2로 받아서, 0과 1로 지정
4) 정규화
5) 파생변수 'BMI' 계산
6) 시각화(GPT의 '전적으로' 도움을 받음)
7) 파일 저장

# 10번_최지수
파일 내용: mental health care 대상자
전처리 순서
1) 라이브러리 호출, 파일 읽기
2) 결측치 처리: 최빈값(문자열도 있기 때문에)으로 대체
3) 불필요한 열 제거: 의미 없는 열, 중복된 내용, 답변이 과도하게 드문 값 제거
4) Age 이상치 처리: 나이 특성 따라 0~100 사이 값만 남김, 이 값들의 평균으로 이 외의 값 대체
5) Gender 재분류: 주석 처리한 코드로 유니크 값 확인, 값마다 male, female, others로 대체
6) 파생 변수: Country 중 의미 있는 값인 United States으로 is_us 열을 대체제로 만듦
7) 변수 분류: object열 중 -> unique 값이 2면 onehot, 2 초과면 label하도록 리스트에 분류
8) Label Encoding + MinMax: *GPT* unique 값 오름차순 정렬 및 숫자 배정, 전체 0~1 minmax 정규화
9) One-hot Encoding: 자동 onehot encoding, Gender열도
10) Age 정규화: Age열 minmax 정규화
11) 저장: 최종 csv파일로 저장





## 문제 4번
# 🏥 Hospital Appointment Data Preprocessing

고객 예약 데이터를 정제하고 머신러닝 분석에 최적화된 형태로 가공하는 전처리 파이프라인입니다.  
특히, 예약 노쇼(No-show) 예측 모델 구축 및 데이터 탐색을 위한 사전 준비 작업을 목적으로 하고 있습니다. 

---

## 프로젝트 개요

**목적**  
- 중복 데이터 제거
- 날짜 데이터 표준화
- 수치형 데이터 변환 및 이상치 제거
- 범주형 데이터 이진화
- 지역 정보 정제 및 희귀 지역 구분

**활용 예시**  
- 환자 노쇼 예측 모델
- 예약 패턴 분석
- 환자 세분화 및 리텐션 전략 수립

---

## 🛠 기능 요약

| 기능 | 상세 설명 |
|:--|:--|
| AppointmentID 중복 제거 | 예약 건별로 고유 데이터만 유지 |
| 날짜 변환 | ScheduledDay, AppointmentDay를 `datetime` 타입으로 변환 |
| 나이 데이터 클린징 | Age를 수치형으로 변환, 0~100세 사이만 유지 |
| 방문 여부 이진화 | 'No-show'를 0(No), 1(Yes)로 변환 |
| 성별 이진화 | 'Gender'를 0(F), 1(M)로 변환 |
| 지역 데이터 정제 | 특수문자 제거 후, 주요 지역은 1, 희귀 지역은 0으로 이진화 |

---

## 함수 설명

### `full_preprocess(df)`

> 입력된 예약 데이터프레임을 전처리하여, 머신러닝 분석용으로 가공된 데이터프레임을 반환합니다.

- **중복 제거**: `AppointmentID` 기준 중복 제거
- **날짜 변환**: `ScheduledDay`, `AppointmentDay`를 `datetime` 변환
- **나이 필터링**: 0~100세 범위 외 데이터 제거
- **'No-show' 이진화**: No=0, Yes=1
- **'Gender' 이진화**: F=0, M=1
- **Neighbourhood 전처리**:
  - 특수문자 제거
  - 등장 빈도 기준 주요 지역(1) vs 희귀 지역(0) 구분

---

## 코드 사용법

```python
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('your_appointment_data.csv')

# 전처리 함수 실행
df_clean = full_preprocess(df)

# 결과 확인
print(df_clean.head())
```
---

## 담당자 

-곽주하(Gwak juha)

---

## 문제 6번
# 🛒 Customer Segmentation Pipeline

고객 구매 이력을 기반으로 RFM 분석(Recency, Frequency, Monetary)을 수행하고,  
이를 통해 고객을 VIP, 일반(General), 휴면(Dormant) 그룹으로 분류하는 전체 파이프라인입니다.  
머신러닝 모델 학습을 위한 데이터셋을 준비하는 것을 목적으로 하고 있습니다. 

---

## 프로젝트 개요

- **구매 데이터 정제 및 전처리**  
- **고객별 RFM 점수 계산 및 조합**
- **VIP, 일반, 휴면 고객 세그먼트 분류**
- **머신러닝 모델 학습용 데이터셋 자동 생성**

---

## 파일 구성

| 파일명 | 설명 |
|:--|:--|
| `full_preprocess_customer_segmentation(df)` | 구매 데이터 전처리 및 고객별 집계 |
| `rfm_segmentation(customer_df)` | 고객별 RFM 점수 부여 |
| `assign_customer_segment(customer_df)` | 고객 세그먼트 분류 |
| `full_customer_segmentation_pipeline(df)` | 위 전체 과정을 통합한 파이프라인 |

---

## 🛠 주요 기능 상세 설명

### 1. 데이터 전처리: `full_preprocess_customer_segmentation(df)`

- **이상치 제거**: 수량(`Quantity`) 및 단가(`Price`)가 0 이하인 데이터 제거
- **결측치 제거**: `Customer ID`가 없는 데이터 제거
- **날짜 변환**: `InvoiceDate`를 `datetime` 타입으로 변환
- **파생 변수 생성**:
  - `Month`: 구매월
  - `Weekday`: 구매 요일 (0=월요일, 6=일요일)
  - `TotalPrice`: 구매수량 × 단가
- **고객별 집계**:
  - 가장 최근 구매일(`LastPurchaseDate`)
  - 총 구매 건수(`PurchaseCount`)
  - 총 구매 수량(`TotalQuantity`)
  - 총 구매 금액(`TotalSpent`)
- **Recency 계산**:
  - 기준일(2011-12-10) 기준으로 마지막 구매일과의 차이(일수)

---

### 2. RFM 점수 부여: `rfm_segmentation(customer_df)`

- **Recency(최근성)**:
  - 최근에 구매했을수록 높은 점수(5점)
- **Frequency(구매 빈도)**:
  - 구매 횟수가 많을수록 높은 점수(5점)
- **Monetary(구매 금액)**:
  - 지출 금액이 클수록 높은 점수(5점)
- **RFM_Score 조합**:
  - R, F, M 점수를 문자열로 연결하여 `'545'`, `'212'` 등으로 저장

---

### 3. 고객 세그먼트 분류: `assign_customer_segment(customer_df)`

- **VIP 고객**:
  - RFM_Score가 '555', '554', '545', '544' 중 하나
- **휴면 고객(Dormant)**:
  - RFM_Score가 '111', '112', '121', '211', '212' 중 하나
- **일반 고객(General)**:
  - 나머지 고객

---

### 4. 전체 통합 파이프라인: `full_customer_segmentation_pipeline(df)`

- 전처리 ➔ RFM 점수 부여 ➔ 세그먼트 분류를 한 번에 수행
- 최종적으로 고객별 `CustomerID`, `RFM_Score`, `Segment` 정보를 포함하는 DataFrame 반환

---

## 실행 방법

```python
# 데이터 불러오기
import pandas as pd
df = pd.read_csv('your_data.csv')

# 전체 파이프라인 실행
segmented_customers = full_customer_segmentation_pipeline(df)

# 결과 확인
print(segmented_customers.head())
```

---
## 담당자

곽주하(Gwak juha)
