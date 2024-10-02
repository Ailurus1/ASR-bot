from pydantic import BaseModel
from typing import Any

from fastapi import APIRouter

class Request(BaseModel):
    user_id: str
    data: float

class Response(BaseModel):
    user_id: str
    content: Any


router = APIRouter()

# curl -X POST "http://127.0.0.1:8080/asr" -H "Content-Type: application/json" -d '{"user_id": "1233", "data": 10}'
@router.post("/asr")
def request_completion(request: Request) -> Response:

    return Response(user_id=request.user_id, content=request.data) 
    