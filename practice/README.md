Midterm_3_practice
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
