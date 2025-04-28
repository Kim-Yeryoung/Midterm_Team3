# Midterm_Team3
## 문제 1번
# 성인 인구 소득 데이터셋 전처리

성인 소득 데이터(Adult Census Income)를 정제하고 머신러닝 분석에 최적화된 형태로 가공하는 전처리 파이프라인입니다. 특히 연간소득을 타겟팅하여 인구통계 정보를 기반으로 예측 모델 구축을 위한 데이터처리를 목표로 삼았습니다.

#### 전처리 설명:
##### 결측치 처리: ?로 처리 되어있는 데이터를 NaN값으로 변환 후 수치형은 중앙값, 범주형은 최빈값으로 처리
##### 이상치 제거: IQR을 사용하여 quatile 1 and 3 에서 부터 IQR의 1.5배 이상 넘어가는 값들을 이상치라 판단, 제거
##### 범주형 엔코딩: 성별을 제외한 모든 오브젝트, 카테고리 범주의 데이터 타입 컬럼을 LabelEncoder으로 처리;
###### 성별 컬럼은 onehot encoding사용 ∵컬럼내 데이터의 variation이 적음(Male/Female)
##### 파생변수 생성: 나이를 30마다 구간을 나눠 카테고리화
##### 정규화: StandardScaler()를 사용하여 모든 컬럼에 대한 정규화 스케일링 실행
##### 전처리 파이프라인 실행: 타겟 컬럼을 제외한 모든 컬럼에 대해 전처리 실행

#### 보완점:
##### 조금 더 column-wise 접근했으면 더 좋았겠다 라는 생각이 듭니다. 이상치 제거 할때도 한 원소에 이상치 발견시 그 row를 제거하는 방향으로 했었는데 이를 보완하여 결측값 설정 -> 결측값 처리 방식으로 하면 모델학습 정확도가 더 올라가지 않을까 싶습니다. 또한 feature 만들떄 시간을 더 투자하여 더 분석적으로 의미있는 컬럼을 만들어 분석보델 성능 향상 시도 해보면 좋겠습니다. 


## 문제 2번
# 🏦 Credit Card Customer Data Preprocessing

신용카드 고객 데이터를 정제하고 머신러닝 분석에 최적화된 형태로 가공하는 전처리 파이프라인입니다.  
특히, 신용 리스크 분석이나 고객 세분화를 위한 모델 구축 및 데이터 탐색을 목적으로 하고 있습니다.

---

## 📋 프로젝트 개요

### 목적
- 중복 데이터 제거
- 결측치 채우기
- 수치형 이상치 제거
- 범주형 데이터 인코딩 (Label Encoding, One-hot Encoding)
- 불필요한 컬럼 제거

### 활용 예시
- 신용카드 한도 예측 모델
- 고객 연체 패턴 분석
- 고객군 분류 및 마케팅 전략 수립

---

## ⚙️ 사용 방법

```python
import pandas as pd

# 파일 경로 설정
input_file = "C:/Users/kimye/Desktop/2_Card.csv"

# 전처리 함수 호출
df = some_function(input_file)

🛠️ 전처리 과정
- 파일 읽기: pandas로 CSV 파일 불러오기
- 결측치 처리: 수치형은 중앙값, 범주형은 최빈값으로 채움
- 이상치 제거: 수치형 데이터에서 IQR 기준 3배 초과 값 제거
- 중복 제거: '업체명', '주업종', '사업자등록번호' 기준 중복 삭제
- 열 삭제: 50% 이상 결측 열과 'Transaction Date' 삭제
- 범주형 인코딩: 다중 범주는 Label Encoding, 이진 범주는 One-hot Encoding

🚨 참고사항: 실행 시 발생한 문제
- `some_function()` 안에서 전처리가 끝나지 않고 df만 반환되어 추가 작업이 필요했습니다.
- 저장(`to_csv`)도 자동으로 되지 않아 직접 실행해야 했습니다.

해결 방법
- 전처리와 저장까지 `some_function()` 안에서 모두 처리하거나,
- 바깥에서 순서대로 직접 처리해야 합니다.

```
## 문제 3번


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


## 문제 5번
## 문제 6번
## 문제 7번
#Heart Disease UCI Dataset
#### 전처리 설명:
##### 결측치 부재를 df.isnull().sum()으로 확인
##### 이상치 제거: IQR을 사용하여 quatile 1 and 3 에서 부터 IQR의 1.5배 이상 넘어가는 값들을 이상치라 판단, 제거
##### df.info()를 통해 범주형 데이터를 포함하는 컬럼이 존재 하지 않음을 확인
##### 파생변수 생성: 나이를 40마다 구간을 나눠 카테고리화
###### tresbps(안정시 심박수)를 사용하여 평균 130이 넘을시 고혈압으로 변수 생성
###### col(콜레스테롤 수치)가 240을 넘을시 콜레스테롤 수치가 높음으로 변수 생성
###### thalach(운동 후 최고 심박수)가 100 보다 작을시 low_max_hr 변수 생성
###### 위 세 항목과 fbs(공복혈당), 그리고 oldpeak(운동 중 ST 저하량)이 threshold(2)를 넘으면 포함하여 총 다섯가지의 분석 데이터를 가지고 새로운 컬럼 위험도(점수)를 생성함
##### 정규화: StandardScaler()를 사용하여 모든 컬럼에 대한 정규화 스케일링 실행
##### 전처리 파이프라인 실행. 모든 컬럼에 대해 전처리 실행

##### 보완점:
파생변수를 생성할때 실질적인 risk score와 비슷하게 항목별 가중치를 주어 실제와 더 가깝게 만들었으면 더 좋았겠다 라는 생각이 듭니다. 또한 나이나 과거 심장병 이력 또한 risk score에 영향을 어느정도 줄텐데 이를 반영하기엔 시간적으로 부족함을 느껴서 이후에 이를 보완해서 더 정량적인 risk score 변수를 생성하고 모델이 이를 학습해서 심장병의 유무 예측, 더 나아가 심장병이 걸릴 확률등을 예측할 수 있게 만들면 이상적일것 같습니다
