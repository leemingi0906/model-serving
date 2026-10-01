"""
4. 응답과 에러 처리 실습

- dict를 그냥 return하면 FastAPI가 알아서 JSON으로 직렬화
- raise HTTPException(status_code, detail) 로 의도적인 4xx 응답
- 잡지 못한 예외는 500

실행 (이 파일이 있는 폴더에서):
    uvicorn ex4_response_error:app --reload

확인:
    http://localhost:8000/docs 에서 GET /items/{item_id}를 아래 세 가지로 호출해보세요.
    - item_id = 1   -> 200, {"item_id": 1, "name": "사과"}
    - item_id = 999 -> 404, {"detail": "item_id=999를 찾을 수 없습니다"}
    - item_id = -1  -> 500 (서버 콘솔에 ZeroDivisionError 트레이스백이 찍힙니다)
"""
from fastapi import FastAPI, HTTPException

app = FastAPI(title="4. 응답과 에러 처리 실습")

ITEMS = {1: "사과", 2: "바나나"}


@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id < 0:
        # 일부러 아무 예외 처리도 하지 않습니다.
        # -> 이 줄에서 예외가 발생하고, FastAPI가 잡지 못한 예외는 500이 됩니다.
        return 1 / 0  # ZeroDivisionError -> 500 Internal Server Error

    if item_id not in ITEMS:
        # 의도적으로 4xx를 내려주고 싶을 때는 이렇게 직접 raise합니다.
        raise HTTPException(status_code=404, detail=f"item_id={item_id}를 찾을 수 없습니다")

    # 그냥 dict를 return하면 FastAPI가 알아서 JSON으로 직렬화합니다.
    return {"item_id": item_id, "name": ITEMS[item_id]}
