"""
app.py —— 材料性质预测 Agent 的 FastAPI 服务
=========================================================
对外接口：POST /chat
请求：{"message": "GaN 的带隙是多少", "history": [{"role": "user", "content": "..."}]}
响应：{"answer": "..."}

启动：
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import agent
import generator
import predict as P

_lock = threading.Lock()
_client = None


@asynccontextmanager
async def lifespan(app):
    """启动时预加载结构生成模型、GNN 和 DeepSeek 客户端。"""
    global _client
    generator._load()
    P.get_model()
    _client = agent.get_client()
    yield


app = FastAPI(title="材料性质预测 Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)


@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
    messages += [
        {"role": m.role, "content": m.content}
        for m in req.history[-8:]
        if m.role in ("user", "assistant")
    ]
    messages.append({"role": "user", "content": req.message})
    with _lock:
        answer = agent.run_agent_loop(_client, messages)
    return {"answer": answer}
