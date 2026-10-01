#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : architect@sk.com, audit@korea.ac.kr 임성열 Ph.D.

# FastAPI 기본 문법 실습 코드

`FastAPI_기본 문법_이해` 문서의 6개 그룹을 각각 단독으로 돌려볼 수 있는 실습 코드 6개와,
그 6개를 모두 연결한 통합 실습 1개(총 7개)입니다. HAIC 프로젝트(`project/`)와는 별개의,
아주 작은 "메모(Note) API" 도메인으로 만들어서 TensorFlow·MLflow 없이 즉시 실행됩니다.

## 준비

```bash
python3.11 -m venv fastapi && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1~4번과 6번은 http://localhost:8000/docs(Swagger UI)에서 해당 엔드포인트를 펼쳐 Try it out으로 직접 호출해보고 응답을 확인하는 방식입니다.

5번과 7번은 정적 파일 서빙이 포함돼 있어서 조금 다릅니다. 5번은 Swagger 대신 http://localhost:8000/에 접속해 static/index.html이 뜨는지, /ping이 정적 파일에 가로채이지 않고 응답하는지를 브라우저로 직접 확인하고, 서버 콘솔에 [startup] 로그가 찍히는지도 함께 봅니다. 7번은 /의 대시보드 화면과 /docs에서의 API 호출을 둘 다 확인하는 방식입니다.

## 실습 목록

```
fastapi_practice/
├── ex1_router_structure.py     # 1. 앱/라우터 구조
│                                #    uvicorn ex1_router_structure:app --reload
├── ex2_request_methods.py      # 2. 요청을 받는 3가지 방법
│                                #    uvicorn ex2_request_methods:app --reload
├── ex3_pydantic_schema.py      # 3. Pydantic으로 스키마 정의
│                                #    uvicorn ex3_pydantic_schema:app --reload
├── ex4_response_error.py       # 4. 응답과 에러 처리
│                                #    uvicorn ex4_response_error:app --reload
├── ex5_lifecycle_static/        # 5. 앱 생명주기 & 정적 파일
│   ├── main.py                  #    cd ex5_lifecycle_static && uvicorn main:app --reload
│   └── static/
│       └── index.html
├── ex6_run_docs.py              # 6. 실행과 자동 문서
│                                #    uvicorn ex6_run_docs:app --reload
└── ex7_integrated/               # 1~6 전부 연결한 통합 실습
    ├── main.py                   #    cd ex7_integrated && uvicorn main:app --reload
    ├── schemas.py
    ├── routers/
    │   ├── health.py
    │   └── notes.py
    └── static/
        └── index.html
```

각 파일 맨 위 docstring에 실행 명령과 `/docs`에서 무엇을 눌러 확인하면 되는지 체크하세요.

한 번에 하나씩만 8000번 포트를 쓰므로, 다음 실습으로 넘어가기 전에 `Ctrl+C`로 이전 서버를
멈추고 진행하세요. 포트를 바꾸고 싶다면 `--port 8001`처럼 옵션을 추가하면 됩니다.

## 순서 추천

1~6번을 순서대로 돌려보며 각 절의 문법이 실제로 어떻게 동작하는지 확인한 뒤, 7번
(`ex7_integrated/`)에서 그 6가지가 하나의 앱 안에 함께 있을 때 파일이 어떻게 나뉘는지
확인하세요. `ex7_integrated/`의 파일 구조는 `project/serving_app/`과 1:1로 대응됩니다
(README의 표 참고). 이 대응 관계를 이해하고 나면 HAIC 실습 코드를 열었을 때 "어디에 뭐가
있는지" 훨씬 빠르게 파악할 수 있습니다.
