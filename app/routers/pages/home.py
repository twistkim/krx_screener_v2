from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """홈 화면.

    - main.py에서 request.app.state.templates(Jinja2Templates)가 세팅되어 있어야 함.
    - page_js/page_css 를 넘겨서 base.html에서 페이지별로 필요한 정적 파일을 include 할 수 있게 함.
    """
    templates = request.app.state.templates  # type: ignore[attr-defined]

    # 간단한 캐시 버스트 (개발 중 브라우저 캐시 때문에 JS 변경이 반영 안 되는 경우 대비)
    # 운영에서는 git commit hash 같은 걸로 바꿔도 됨.
    v = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    return templates.TemplateResponse(
        "pages/home.html",
        {
            "request": request,
            "title": "KRX Screener V2",
            # base.html에서 아래 변수를 사용해 <script>/<link>를 붙일 수 있게 함
            "page_js": f"js/pages/home.js?v={v}",
            "page_css": None,
        },
    )