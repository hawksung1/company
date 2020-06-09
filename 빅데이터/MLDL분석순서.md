# MLDL분석 순서

1. 프로젝트 설정
    프로젝트 자원 설정
2. 프로젝트 멤버 설정
    관리자
        프로젝트 자원 설정 및 분석환경에 대한 설정. sudo
    분석가
        본인의 분석환경에만 접근 가능
    리뷰어
        분석 현황 조회
3. 데이터셋 등록
    데이터가 크다면 pyHive, pySpark 등을 이용해 hive 서버로 데이터를 불러오도록 처리
    공용 data는  shared에 보관
4. 분석환경 설정
    **PSP**
        Python Spark
    PML
        Python machine learning
    **RSP**
        r studio spark
    RML
        r studio machine learning
5. 개발 및 분석
    패키지 install
    이후 분석환경 이미지 저장
6. model 등록
    job scheduling을 통해 kpi 표시
7. job scheduling
    보다 많은 리소스 사용 가능
8. 헉숩 결과 및 평가 지표 조회
    KPI 지표에 맞춰 결과 조회