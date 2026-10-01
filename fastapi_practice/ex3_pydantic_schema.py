"""
3. Pydantic으로 스키마 정의 실습

- class X(BaseModel) 기본형
- Field(..., gt=0, min_length=, max_length=) 로 제약 걸기
- 중첩 모델 (list[Tag])
- response_model 로 응답도 같은 방식으로 검증

실행 (이 파일이 있는 폴더에서):
    uvicorn ex3_pydantic_schema:app --reload

확인:
    http://localhost:8000/docs 에서 POST /items 를 두 번 호출해보세요.
    1) 유효한 값 : {"title": "노트북", "price": 1500000, "tags": [{"name": "전자기기"}]}
       -> 200 응답
    2) 일부러 잘못된 값 : {"title": "", "price": -1, "tags": []}
       -> 422 응답. 어떤 필드가 왜 걸렸는지 loc/msg를 읽어보세요.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="3. Pydantic 스키마 정의 실습")


# 중첩되는 모델도 그냥 BaseModel을 상속한 클래스로 정의합니다.
class Tag(BaseModel):
    name: str = Field(..., min_length=1, description="태그 이름 (1글자 이상)")


class ItemCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=50, description="1~50자")
    price: float = Field(..., gt=0, description="0보다 커야 함")
    tags: list[Tag] = Field(..., min_length=1, description="태그 1개 이상 필요")


# 응답 전용 스키마 - 요청 스키마(ItemCreate)와 다른 모양이어도 됩니다.
class ItemResponse(BaseModel):
    id: int
    title: str
    price: float
    tag_count: int


@app.post("/items", response_model=ItemResponse)
def create_item(item: ItemCreate):
    # response_model을 지정하면, 여기서 무엇을 return하든
    # FastAPI가 ItemResponse 모양대로 다시 한 번 검증·직렬화합니다.
    return ItemResponse(id=1, title=item.title, price=item.price, tag_count=len(item.tags))
