"""
K-REACH Advisor — FastAPI Backend
"""
import os, json, io, bcrypt
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
import anthropic
from supabase import create_client

# ── Init ───────────────────────────────────────────────────────
app = FastAPI(title="K-REACH Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 실제 도메인으로 교체
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
claude    = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
JWT_SECRET = os.environ["JWT_SECRET"]
ADMIN_KEY  = os.environ.get("ADMIN_KEY", "change-me")

# ── K-REACH 시스템 프롬프트 ─────────────────────────────────────
KREACH_SYSTEM = """당신은 대한민국 화학물질 규제(화평법·화관법·K-REACH) 전문 컨설턴트입니다.
핵심 규제 지식:
- 기존물질: 연간 1톤 이상 제조/수입 시 등록 의무 (톤수대별 유예기간 적용)
- 신규물질: 0.1톤 이상 제조/수입 전 등록 필요, 0.1톤 미만 신고
- 2025.01.01: 신규물질 신고기준 0.1→1톤 미만으로 상향
- 2025.08.07: 유독물질 3분류(인체급성/인체만성/생태유해성) 시행, LOC 새 양식
- 2026.07.01 마감: SDS 15항(규제정보) 전면 개정 제출
- 면제대상: 방사성물질, 의약품, 농약, 비료, 화장품, 식품첨가물 등
- 화관법: 유해화학물질 취급시설 설치·운영 기준, 사고예방계획서 작성 의무
- 공동등록: 동일물질 등록자 협의체 구성 가능
- SDS: 16개 항목 한국어 작성, 수입업체는 하위사용자 전달 의무
한국어 마크다운(###, **, 불릿)으로 실용적으로 답하세요. 불확실한 내용은 명시하세요."""

# ── Auth 헬퍼 ──────────────────────────────────────────────────
def make_token(user_id: str) -> str:
    exp = datetime.utcnow() + timedelta(days=30)
    return jwt.encode({"sub": user_id, "exp": exp}, JWT_SECRET, algorithm="HS256")

def current_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "인증이 필요합니다.")
    try:
        payload = jwt.decode(authorization.split()[1], JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except Exception:
        raise HTTPException(401, "유효하지 않은 토큰입니다.")

# ── Pydantic Models ────────────────────────────────────────────
class RegisterReq(BaseModel):
    email: str
    password: str
    company_name: str
    company_type: str
    industry: str
    company_size: str

class LoginReq(BaseModel):
    email: str
    password: str

class ProductSubstance(BaseModel):
    name: str
    cas: str | None = None
    pct_min: float | None = None
    pct_max: float | None = None
    confidence: str = "medium"

class Product(BaseModel):
    name: str
    annual_ton: float
    substances: list[ProductSubstance]

class AnalysisReq(BaseModel):
    products: list[Product]

class ChatReq(BaseModel):
    question: str

# ── 롤업 계산 ──────────────────────────────────────────────────
def build_rollup(products: list[Product]) -> list[dict]:
    rollup: dict[str, dict] = {}
    for prod in products:
        ton = prod.annual_ton or 0
        for s in prod.substances:
            key = (s.cas or s.name).strip().lower()
            pct  = s.pct_max or s.pct_min or 100
            subst_ton = ton * (pct / 100)
            if key not in rollup:
                rollup[key] = {"name": s.name, "cas": s.cas, "total_ton": 0.0, "products": []}
            rollup[key]["total_ton"] += subst_ton
            rollup[key]["products"].append(prod.name)
    return sorted(rollup.values(), key=lambda x: -x["total_ton"])

def verdict(ton: float) -> dict:
    if ton < 0.1:   return {"status": "소량·면제 검토", "cls": "exempt"}
    if ton < 1:     return {"status": "신고 필요",       "cls": "notify"}
    if ton < 10:    return {"status": "등록 필요",       "cls": "reg"}
    if ton < 100:   return {"status": "등록 필요",       "cls": "reg"}
    if ton < 1000:  return {"status": "등록 필요(고톤수)","cls": "reg"}
    return              {"status": "등록 필요(최고톤수)","cls": "reg"}

# ── 규제 컨텍스트 가져오기 ─────────────────────────────────────
def get_reg_context() -> str:
    try:
        rows = supabase.table("regulations")\
            .select("title,content,effective_date")\
            .order("effective_date", desc=True).limit(10).execute()
        if rows.data:
            return "\n".join(f"- [{r['effective_date']}] {r['title']}: {r['content'][:150]}"
                             for r in rows.data)
    except Exception:
        pass
    return "규제 데이터 없음 (기본 학습 데이터 기반 분석)"

# ══════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════
@app.post("/auth/register")
async def register(req: RegisterReq):
    existing = supabase.table("users").select("id").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(400, "이미 등록된 이메일입니다.")
    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    row = supabase.table("users").insert({
        "email": req.email, "password_hash": pw_hash,
        "company_name": req.company_name, "company_type": req.company_type,
        "industry": req.industry, "company_size": req.company_size,
    }).execute().data[0]
    return {"token": make_token(row["id"]),
            "user": {k: row[k] for k in ["id","email","company_name","company_type","industry","company_size"]}}

@app.post("/auth/login")
async def login(req: LoginReq):
    rows = supabase.table("users").select("*").eq("email", req.email).execute()
    if not rows.data:
        raise HTTPException(401, "이메일 또는 비밀번호가 올바르지 않습니다.")
    user = rows.data[0]
    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(401, "이메일 또는 비밀번호가 올바르지 않습니다.")
    return {"token": make_token(user["id"]),
            "user": {k: user[k] for k in ["id","email","company_name","company_type","industry","company_size"]}}

@app.get("/auth/me")
async def me(uid: str = Depends(current_user)):
    row = supabase.table("users")\
        .select("id,email,company_name,company_type,industry,company_size,created_at")\
        .eq("id", uid).execute()
    if not row.data:
        raise HTTPException(404)
    return row.data[0]

# ══════════════════════════════════════════════════════════════
# TDS UPLOAD & PARSE
# ══════════════════════════════════════════════════════════════
@app.post("/upload/tds")
async def upload_tds(file: UploadFile = File(...), uid: str = Depends(current_user)):
    content = await file.read()

    # 텍스트 추출
    text = ""
    if file.filename.lower().endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            text = content.decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        return {"product_name": file.filename, "substances": [], "notes": "텍스트를 추출할 수 없습니다."}

    prompt = f"""다음 TDS/SDS에서 화학 성분을 추출하세요.

문서:
\"\"\"
{text[:3500]}
\"\"\"

JSON만 반환:
{{"product_name":"제품명","substances":[{{"name":"물질명","cas":"CAS번호","pct_min":최소,"pct_max":최대,"confidence":"high|medium|low"}}],"notes":"특이사항"}}"""

    resp = claude.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system="화학 TDS/SDS 파서. JSON으로만 응답.",
        messages=[{"role":"user","content":prompt}]
    )
    raw = resp.content[0].text.replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"product_name": file.filename, "substances": [], "notes": "파싱 실패 — 수동 입력 필요"}

# ══════════════════════════════════════════════════════════════
# ANALYSES
# ══════════════════════════════════════════════════════════════
@app.post("/analyses")
async def create_analysis(req: AnalysisReq, uid: str = Depends(current_user)):
    rollup = build_rollup(req.products)
    for r in rollup:
        r.update(verdict(r["total_ton"]))

    # 유저 정보
    user_row = supabase.table("users")\
        .select("company_name,company_type,industry,company_size")\
        .eq("id", uid).execute()
    user = user_row.data[0] if user_row.data else {}

    reg_ctx = get_reg_context()

    subst_lines = "\n".join(
        f"- {r['name']} (CAS:{r.get('cas','?')}): 합산 {r['total_ton']:.2f}톤 → {r['status']}"
        for r in rollup
    )

    base_prompt = f"""회사: {user.get('company_name','')} ({user.get('company_type','')} / {user.get('industry','')} / {user.get('company_size','')})
제품 수: {len(req.products)}개

물질별 합산:
{subst_lines}

최신 규제 현황:
{reg_ctx}

각 물질의 K-REACH 등록/신고 의무, 근거 조문, SDS 개정 필요 여부를 분석하세요."""

    # 가이드
    g = claude.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system=KREACH_SYSTEM, messages=[{"role":"user","content":base_prompt}]
    )
    guidance = g.content[0].text

    # 액션플랜
    a = claude.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system=KREACH_SYSTEM,
        messages=[
            {"role":"user","content":base_prompt},
            {"role":"assistant","content":guidance},
            {"role":"user","content":"월별 액션 플랜(즉시/1개월/6개월), 준비 서류, 예상 비용을 작성하세요."}
        ]
    )
    action_plan = a.content[0].text

    # DB 저장
    saved = supabase.table("analyses").insert({
        "user_id": uid,
        "product_count": len(req.products),
        "substance_count": len(rollup),
        "products_json": json.dumps([p.dict() for p in req.products], ensure_ascii=False),
        "rollup_json": json.dumps(rollup, ensure_ascii=False),
        "guidance": guidance,
        "action_plan": action_plan,
        "status": "completed",
    }).execute()

    return {
        "id": saved.data[0]["id"],
        "rollup": rollup,
        "guidance": guidance,
        "action_plan": action_plan,
    }

@app.get("/analyses")
async def list_analyses(uid: str = Depends(current_user)):
    rows = supabase.table("analyses")\
        .select("id,product_count,substance_count,status,created_at")\
        .eq("user_id", uid).order("created_at", desc=True).execute()
    return rows.data or []

@app.get("/analyses/{aid}")
async def get_analysis(aid: str, uid: str = Depends(current_user)):
    rows = supabase.table("analyses").select("*").eq("id", aid).eq("user_id", uid).execute()
    if not rows.data:
        raise HTTPException(404, "분석을 찾을 수 없습니다.")
    d = rows.data[0]
    d["rollup"] = json.loads(d.get("rollup_json") or "[]")
    return d

@app.post("/analyses/{aid}/chat")
async def analysis_chat(aid: str, req: ChatReq, uid: str = Depends(current_user)):
    rows = supabase.table("analyses")\
        .select("rollup_json,guidance").eq("id", aid).eq("user_id", uid).execute()
    if not rows.data:
        raise HTTPException(404)
    ctx = rows.data[0]
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system=KREACH_SYSTEM,
        messages=[{"role":"user","content":
            f"분석 컨텍스트:\n{ctx['rollup_json']}\n\n기존 가이드:\n{ctx['guidance']}\n\n질문: {req.question}"}]
    )
    return {"answer": resp.content[0].text}

# ══════════════════════════════════════════════════════════════
# REGULATIONS
# ══════════════════════════════════════════════════════════════
@app.get("/regulations")
async def get_regulations():
    rows = supabase.table("regulations")\
        .select("*").order("effective_date", desc=True).execute()
    return rows.data or []

@app.post("/admin/update-regulations")
async def update_regulations(admin_key: str = Header(None)):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Forbidden")
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system="K-REACH·화평법·화관법 규제 전문가. 최신 개정사항을 JSON 배열로만 응답.",
        messages=[{"role":"user","content":
            """2024~2026년 K-REACH(화평법)·화관법 주요 개정/시행 사항을 JSON 배열로 정리:
[{"title":"개정명","content":"핵심 내용 2~3줄","effective_date":"YYYY-MM-DD","source":"환경부 또는 법령명"}]
반드시 JSON만 반환."""}],
        tools=[{"type":"web_search_20250305","name":"web_search"}]
    )
    text = "".join(b.text for b in resp.content if hasattr(b,"text"))
    text = text.replace("```json","").replace("```","").strip()
    try:
        regs = json.loads(text)
        count = 0
        for r in regs:
            supabase.table("regulations").upsert({
                "title": r.get("title",""),
                "content": r.get("content",""),
                "effective_date": r.get("effective_date","2025-01-01"),
                "source_url": r.get("source",""),
                "updated_at": datetime.utcnow().isoformat(),
            }, on_conflict="title").execute()
            count += 1
        return {"updated": count, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(500, f"파싱 실패: {e} | raw: {text[:300]}")

# ══════════════════════════════════════════════════════════════
# ADMIN — 모든 엔드포인트는 admin-key 헤더 인증
# ══════════════════════════════════════════════════════════════
def check_admin(admin_key: str = Header(None)):
    if not admin_key or admin_key != ADMIN_KEY:
        raise HTTPException(403, "관리자 권한이 없습니다.")

# ── 관리자 로그인 ──────────────────────────────────────────────
class AdminLoginReq(BaseModel):
    key: str

@app.post("/admin/login")
async def admin_login(req: AdminLoginReq):
    if req.key != ADMIN_KEY:
        raise HTTPException(403, "관리자 키가 올바르지 않습니다.")
    token = jwt.encode(
        {"sub": "admin", "role": "admin", "exp": datetime.utcnow() + timedelta(days=1)},
        JWT_SECRET, algorithm="HS256"
    )
    return {"token": token}

def check_admin_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "인증이 필요합니다.")
    try:
        payload = jwt.decode(authorization.split()[1], JWT_SECRET, algorithms=["HS256"])
        if payload.get("role") != "admin":
            raise HTTPException(403, "관리자 권한이 없습니다.")
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "토큰이 만료되었습니다.")
    except Exception:
        raise HTTPException(401, "유효하지 않은 토큰입니다.")

# ── 대시보드 통계 ──────────────────────────────────────────────
@app.get("/admin/stats")
async def admin_stats(_=Depends(check_admin_token)):
    users     = supabase.table("users").select("id,created_at").execute()
    analyses  = supabase.table("analyses").select("id,created_at,substance_count").execute()
    regs      = supabase.table("regulations").select("id").execute()

    now   = datetime.utcnow()
    week  = (now - timedelta(days=7)).isoformat()
    month = (now - timedelta(days=30)).isoformat()

    new_users_week  = sum(1 for u in (users.data or []) if u["created_at"] >= week)
    new_anal_week   = sum(1 for a in (analyses.data or []) if a["created_at"] >= week)
    new_anal_month  = sum(1 for a in (analyses.data or []) if a["created_at"] >= month)
    total_subst     = sum(a.get("substance_count",0) for a in (analyses.data or []))

    return {
        "total_users":      len(users.data or []),
        "total_analyses":   len(analyses.data or []),
        "total_regulations":len(regs.data or []),
        "new_users_week":   new_users_week,
        "new_analyses_week":new_anal_week,
        "new_analyses_month":new_anal_month,
        "total_substances_analyzed": total_subst,
    }

# ── 전체 회원 목록 ─────────────────────────────────────────────
@app.get("/admin/users")
async def admin_users(
    page: int = 1, size: int = 20, search: str = "",
    _=Depends(check_admin_token)
):
    offset = (page - 1) * size
    q = supabase.table("users").select(
        "id,email,company_name,company_type,industry,company_size,created_at"
    )
    if search:
        q = q.ilike("email", f"%{search}%")  # 이메일 검색
    rows = q.order("created_at", desc=True).range(offset, offset + size - 1).execute()

    # 각 유저의 분석 건수 추가
    result = []
    for u in (rows.data or []):
        cnt = supabase.table("analyses").select("id", count="exact")\
            .eq("user_id", u["id"]).execute()
        u["analysis_count"] = cnt.count or 0
        result.append(u)

    total = supabase.table("users").select("id", count="exact").execute()
    return {"users": result, "total": total.count or 0, "page": page, "size": size}

# ── 회원 상세 ─────────────────────────────────────────────────
@app.get("/admin/users/{uid}")
async def admin_user_detail(uid: str, _=Depends(check_admin_token)):
    user = supabase.table("users")\
        .select("id,email,company_name,company_type,industry,company_size,created_at")\
        .eq("id", uid).execute()
    if not user.data:
        raise HTTPException(404, "유저를 찾을 수 없습니다.")

    analyses = supabase.table("analyses")\
        .select("id,product_count,substance_count,status,created_at")\
        .eq("user_id", uid).order("created_at", desc=True).execute()

    return {"user": user.data[0], "analyses": analyses.data or []}

# ── 회원 삭제 ─────────────────────────────────────────────────
@app.delete("/admin/users/{uid}")
async def admin_delete_user(uid: str, _=Depends(check_admin_token)):
    supabase.table("analyses").delete().eq("user_id", uid).execute()
    supabase.table("users").delete().eq("id", uid).execute()
    return {"deleted": uid}

# ── 전체 분석 목록 ─────────────────────────────────────────────
@app.get("/admin/analyses")
async def admin_analyses(
    page: int = 1, size: int = 20,
    _=Depends(check_admin_token)
):
    offset = (page - 1) * size
    rows = supabase.table("analyses")\
        .select("id,user_id,product_count,substance_count,status,created_at")\
        .order("created_at", desc=True).range(offset, offset + size - 1).execute()

    # 이메일·회사명 조인
    result = []
    for a in (rows.data or []):
        u = supabase.table("users").select("email,company_name")\
            .eq("id", a["user_id"]).execute()
        if u.data:
            a["email"] = u.data[0]["email"]
            a["company_name"] = u.data[0]["company_name"]
        result.append(a)

    total = supabase.table("analyses").select("id", count="exact").execute()
    return {"analyses": result, "total": total.count or 0, "page": page, "size": size}

# ── 분석 상세 (관리자) ────────────────────────────────────────
@app.get("/admin/analyses/{aid}")
async def admin_analysis_detail(aid: str, _=Depends(check_admin_token)):
    row = supabase.table("analyses").select("*").eq("id", aid).execute()
    if not row.data:
        raise HTTPException(404)
    d = row.data[0]
    d["rollup"] = json.loads(d.get("rollup_json") or "[]")
    u = supabase.table("users").select("email,company_name,company_type,industry")\
        .eq("id", d["user_id"]).execute()
    if u.data:
        d["user_info"] = u.data[0]
    return d

# ── 분석 삭제 ─────────────────────────────────────────────────
@app.delete("/admin/analyses/{aid}")
async def admin_delete_analysis(aid: str, _=Depends(check_admin_token)):
    supabase.table("analyses").delete().eq("id", aid).execute()
    return {"deleted": aid}

# ── 규제 추가 ─────────────────────────────────────────────────
class RegReq(BaseModel):
    title: str
    content: str
    effective_date: str
    source_url: str = ""

@app.post("/admin/regulations")
async def admin_add_reg(req: RegReq, _=Depends(check_admin_token)):
    row = supabase.table("regulations").insert({
        "title": req.title, "content": req.content,
        "effective_date": req.effective_date, "source_url": req.source_url,
        "updated_at": datetime.utcnow().isoformat(),
    }).execute()
    return row.data[0]

# ── 규제 수정 ─────────────────────────────────────────────────
@app.put("/admin/regulations/{rid}")
async def admin_update_reg(rid: str, req: RegReq, _=Depends(check_admin_token)):
    row = supabase.table("regulations").update({
        "title": req.title, "content": req.content,
        "effective_date": req.effective_date, "source_url": req.source_url,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", rid).execute()
    return row.data[0]

# ── 규제 삭제 ─────────────────────────────────────────────────
@app.delete("/admin/regulations/{rid}")
async def admin_delete_reg(rid: str, _=Depends(check_admin_token)):
    supabase.table("regulations").delete().eq("id", rid).execute()
    return {"deleted": rid}

# ── AI 규제 자동 업데이트 (웹 검색 포함) ──────────────────────
@app.post("/admin/regulations/ai-update")
async def admin_ai_update_regs(_=Depends(check_admin_token)):
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system="K-REACH·화평법·화관법 규제 전문가. 최신 개정사항을 JSON 배열로만 응답.",
        messages=[{"role":"user","content":
            """2024~2026년 K-REACH(화평법)·화관법 주요 개정/시행 사항을 JSON 배열로 정리:
[{"title":"개정명","content":"핵심 내용 2~3줄","effective_date":"YYYY-MM-DD","source":"환경부 또는 법령명"}]
반드시 JSON만 반환."""}],
        tools=[{"type":"web_search_20250305","name":"web_search"}]
    )
    text = "".join(b.text for b in resp.content if hasattr(b,"text"))
    text = text.replace("```json","").replace("```","").strip()
    try:
        regs = json.loads(text)
        count = 0
        for r in regs:
            supabase.table("regulations").upsert({
                "title": r.get("title",""),
                "content": r.get("content",""),
                "effective_date": r.get("effective_date","2025-01-01"),
                "source_url": r.get("source",""),
                "updated_at": datetime.utcnow().isoformat(),
            }, on_conflict="title").execute()
            count += 1
        return {"updated": count, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(500, f"파싱 실패: {e}")

# ── 시스템 헬스체크 ────────────────────────────────────────────
@app.get("/admin/health")
async def admin_health(_=Depends(check_admin_token)):
    db_ok = False
    try:
        supabase.table("users").select("id").limit(1).execute()
        db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "connected" if db_ok else "error",
        "timestamp": datetime.utcnow().isoformat(),
    }
