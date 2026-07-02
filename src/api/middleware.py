"""API 中间件 — 请求日志 + 异常处理"""

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("langgraph-scrm.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 — 记录每个 API 请求的方法、路径、耗时"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} — {duration:.3f}s — {response.status_code}")
        return response
