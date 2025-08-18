# backend/main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

from api.routes import router

app = FastAPI(title="RAG 智能体 API")

# ------------------- CORS -------------------
# 开发时允许前端 Vite 服务器跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- 挂载 API 路由 -------------------
app.include_router(router)

# ------------------- 托管前端静态文件 -------------------
FRONTEND_DIST = Path(__file__).parent.parent / "fontend" / "dist"

if FRONTEND_DIST.exists():
    # 1. 把 dist 目录整体挂到根路径
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    print(f"⚠️ 前端构建目录不存在: {FRONTEND_DIST}")

@app.get("/_health")
def health():
    return {"status": "ok"}

# ------------------- 根路由：返回 index.html -------------------
@app.get("/", response_class=FileResponse)
async def serve_frontend_root():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "前端构建文件未找到，请先运行 npm run build"}

# ------------------- 兜底路由：所有其他路径都返回 index.html（SPA） -------------------
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    file_path = FRONTEND_DIST / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # 否则返回 index.html，由前端路由处理
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "前端构建文件未找到"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)