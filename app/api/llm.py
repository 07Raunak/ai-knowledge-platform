from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_current_user
from app.container import Container
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1/llm", tags=["ai-gateway"])


@router.post("/chat", response_model=ChatResponse, summary="Call an LLM through the centralized AI Gateway")
def chat(req: ChatRequest, user: str = Depends(get_current_user), c: Container = Depends(get_container)):
    result = c.gateway.complete(
        user_id=user,
        purpose="chat",
        messages=[m.model_dump() for m in req.messages],
        system=req.system,
        model=req.model,
        max_tokens=req.max_tokens,
    )
    return ChatResponse(
        model=result.model,
        content=result.text,
        stop_reason=result.stop_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=round(result.latency_ms, 1),
        fallback_used=result.fallback_used,
    )
