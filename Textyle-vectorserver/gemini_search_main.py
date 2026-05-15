"""
Gemini 기반 레퍼런스 이미지 + 텍스트 융합 검색 서버.

흐름:
  1) Gemini Vision 으로 레퍼런스 이미지를 구조화된 속성(JSON)으로 분해
  2) Gemini Text 로 (이미지 속성 + 사용자 요청)을 영문 description 한 줄로 융합
  3) marqo/fashionSigLIP 의 encode_text 로 768-dim 임베딩 생성
  4) Supabase RPC match_clothes 로 코사인 유사도 검색

기존 main.py(8000)와 fashion_main.py 와는 독립된 FastAPI 프로세스(포트 8001).
DB(Supabase) 의 임베딩은 marqo/fashionSigLIP 로 만들어졌으므로 본 파일도 동일 모델을 사용해야
pgvector 차원·의미공간이 일치한다.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client
from PIL import Image
import io
import os
import json
import traceback

import torch
import torch.nn.functional as F
# 🔥 transformers 의 meta-tensor 버그 우회를 위해 open_clip 직접 사용 (main.py 와 동일)
import open_clip
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # google-genai 가 설치되어 있지 않은 환경 대비
    genai = None
    genai_types = None


# ────────────────────────────────────────────────────────────────
# 1. 환경 변수 로드 (main.py 패턴 그대로)
# ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(".env 에 SUPABASE_URL / SUPABASE_KEY 가 설정되어 있어야 합니다.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if (GEMINI_API_KEY and genai is not None)
    else None
)
if gemini_client is None:
    print("⚠️  GEMINI_API_KEY 가 없거나 google-genai 가 미설치되어 Gemini 호출이 비활성화됩니다.")


# ────────────────────────────────────────────────────────────────
# 2. FastAPI & 임베딩 모델
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="TexTyle Gemini Search Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
print(f"AI 모델 로딩 중... (Device: {device}, model: {MODEL_ID})")

# 본 파일은 텍스트 임베딩만 사용하지만, open_clip API 상 preprocess 까지 함께 반환됨
model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_ID)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(MODEL_ID)
print("AI 모델 로딩 완료!")


# ────────────────────────────────────────────────────────────────
# 3. Pydantic 스키마 — Gemini Vision 의 구조화 응답
# ────────────────────────────────────────────────────────────────
class ReferenceImageAttributes(BaseModel):
    category: str = Field(description="garment type, e.g. hoodie, jeans, blazer")
    color: str = Field(description="dominant color in plain English, e.g. navy blue")
    fit: str = Field(description="silhouette/fit, e.g. oversized, slim, regular")
    pattern: str = Field(description="pattern, e.g. solid, striped, graphic print")
    material: str = Field(description="fabric/material if visible, e.g. cotton, denim, leather")
    details: str = Field(description="distinctive details, e.g. kangaroo pocket, ribbed cuffs, zip closure")


def _empty_attrs() -> ReferenceImageAttributes:
    return ReferenceImageAttributes(
        category="", color="", fit="", pattern="", material="", details=""
    )


# ────────────────────────────────────────────────────────────────
# 4. Gemini Vision: 이미지 → 구조화 속성
# ────────────────────────────────────────────────────────────────
VISION_SYSTEM_PROMPT = """
You are a fashion attribute extractor. Look at the single garment in the photo
and extract its attributes as JSON. Use concise English noun phrases. If a
field cannot be determined from the image, return an empty string for that
field. Do NOT invent attributes that are not visible.
""".strip()


def _guess_mime_type(image_bytes: bytes) -> str:
    """PIL 로 이미지 포맷을 추정하여 MIME 타입 반환 (실패 시 jpeg 로 가정)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            fmt = (im.format or "").lower()
        if fmt == "png":
            return "image/png"
        if fmt == "webp":
            return "image/webp"
        if fmt == "gif":
            return "image/gif"
        return "image/jpeg"
    except Exception:
        return "image/jpeg"


def analyze_reference_image(image_bytes: bytes) -> ReferenceImageAttributes:
    """Gemini Vision 호출. 실패 시 빈 속성으로 폴백."""
    if gemini_client is None or genai_types is None:
        return _empty_attrs()

    mime = _guess_mime_type(image_bytes)
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                VISION_SYSTEM_PROMPT,
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime),
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReferenceImageAttributes,
            ),
        )
        # fashion_main.py 와 같은 파싱 패턴: response.parsed → text fallback
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        else:
            data = json.loads(response.text)
        return ReferenceImageAttributes(
            category=str(data.get("category", "") or ""),
            color=str(data.get("color", "") or ""),
            fit=str(data.get("fit", "") or ""),
            pattern=str(data.get("pattern", "") or ""),
            material=str(data.get("material", "") or ""),
            details=str(data.get("details", "") or ""),
        )
    except Exception as exc:
        print(f"[vision] Gemini 실패, 빈 속성으로 폴백: {exc}")
        traceback.print_exc()
        return _empty_attrs()


# ────────────────────────────────────────────────────────────────
# 5. Gemini Text: (속성 + 사용자 요청) → 영문 description 한 줄
# ────────────────────────────────────────────────────────────────
FUSION_SYSTEM_PROMPT = """
You are a fashion search prompt composer.
Inputs:
  - REFERENCE_ATTRS: a JSON object describing a reference garment.
  - USER_QUERY: a user's modification request (Korean or English).

Task:
  Produce ONE concise English description of the target garment the user is
  looking for. If USER_QUERY explicitly changes an attribute (e.g. color,
  fit, pattern, material), OVERRIDE that attribute and keep the rest from
  REFERENCE_ATTRS. If USER_QUERY does not mention an attribute, keep the
  reference value. Output the description only — no quotes, no prefix, no
  JSON. Keep it under 30 words. Example:
    "an oversized navy blue hoodie with a kangaroo pocket and ribbed cuffs, cotton"
""".strip()


def _rule_based_fusion(attrs: ReferenceImageAttributes, user_query: str) -> str:
    """Gemini 사용 불가 시 fallback.

    1) attrs 가 채워져 있으면 attrs + user_query 조합
    2) attrs 가 비어있으면 (Vision 실패/비활성) user_query 만 사용 — 한국어라도 임베딩 시도
    """
    parts = [attrs.fit, attrs.color, attrs.category]
    base = " ".join(p for p in parts if p).strip()
    extras = ", ".join(p for p in [attrs.pattern, attrs.material, attrs.details] if p)
    description = base if not extras else f"{base}, {extras}"

    uq = user_query.strip()
    if not description:
        # attrs 가 전부 비어있는 경우 — user_query 만으로 description 구성
        return f"a fashion item described as: {uq}" if uq else "fashion item"

    if uq:
        description = f"{description}. user request: {uq}"
    return description


def fuse_with_user_query(attrs: ReferenceImageAttributes, user_query: str) -> str:
    if gemini_client is None or genai_types is None:
        return _rule_based_fusion(attrs, user_query)

    try:
        prompt = (
            f"REFERENCE_ATTRS: {json.dumps(attrs.model_dump(), ensure_ascii=False)}\n"
            f"USER_QUERY: {user_query}"
        )
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[FUSION_SYSTEM_PROMPT, prompt],
        )
        text = (getattr(response, "text", "") or "").strip()
        # 안전장치: 빈 응답이면 fallback
        if not text:
            return _rule_based_fusion(attrs, user_query)
        return text
    except Exception as exc:
        print(f"[fusion] Gemini 실패, 규칙 기반 폴백: {exc}")
        traceback.print_exc()
        return _rule_based_fusion(attrs, user_query)


# ────────────────────────────────────────────────────────────────
# 6. 텍스트 → 768-dim 임베딩 (main.py:244-248 패턴)
# ────────────────────────────────────────────────────────────────
def encode_text_to_embedding(text: str) -> list[float]:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens, normalize=True)
    # 이미 normalize=True 이지만 명시적으로 한 번 더 정규화 (안전)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats.squeeze().cpu().tolist()


# ────────────────────────────────────────────────────────────────
# 7. 엔드포인트
# ────────────────────────────────────────────────────────────────
# 텍스트-이미지 크로스모달 코사인 유사도는 image-image 보다 일반적으로 낮음.
# main.py 의 mixed-embedding 임계값(0.45~0.70)을 그대로 쓰면 결과가 비어버릴 수 있어
# 더 낮은 기본값 사용. 운영 중에 환경변수로 조정 가능.
DEFAULT_MATCH_THRESHOLD = float(os.environ.get("AI_SEARCH_THRESHOLD", "0.15"))
DEFAULT_MATCH_COUNT = int(os.environ.get("AI_SEARCH_COUNT", "10"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "gemini_enabled": gemini_client is not None,
        "model": MODEL_ID,
    }


@app.post("/ai_search")
async def ai_search(
    file: UploadFile = File(None),
    query: str = Form(None),
):
    if not file or query is None or not query.strip():
        raise HTTPException(status_code=400, detail="이미지(file)와 요청 텍스트(query)가 모두 필요합니다.")

    try:
        # 1) 이미지 bytes 로드
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="이미지 파일이 비어있습니다.")

        # 2) Gemini Vision → 속성 JSON
        attrs = analyze_reference_image(image_bytes)

        # 3) Gemini Text → 융합된 영문 description
        final_text = fuse_with_user_query(attrs, query)
        print(f"[ai_search] final_text = {final_text!r}")

        # 4) 텍스트 임베딩
        embedding = encode_text_to_embedding(final_text)

        # 5) Supabase RPC (main.py 와 동일 시그니처)
        response = supabase.rpc(
            "match_clothes",
            {
                "query_embedding": embedding,
                "match_threshold": DEFAULT_MATCH_THRESHOLD,
                "match_count": DEFAULT_MATCH_COUNT,
                "filter_main_category": None,
                "filter_sub_category": None,
            },
        ).execute()

        results = response.data or []

        return {
            "message": "Success",
            "gemini_enabled": gemini_client is not None,
            "image_attributes": attrs.model_dump(),
            "final_text": final_text,
            "match_threshold": DEFAULT_MATCH_THRESHOLD,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────────────────────────────────────────────────────
# 8. 실행부
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # 기존 main.py(8000), fashion_main.py 와 충돌하지 않도록 8001 사용.
    uvicorn.run(app, host="0.0.0.0", port=8001)
