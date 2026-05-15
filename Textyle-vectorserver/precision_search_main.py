"""
2-Stage 정밀 검색 서버 (포트 8002).

흐름:
  ⓪ Gemini 후처리      ref_img + 한국어 query → 영문 description 1줄 (final_text)
  ① Stage 1 (Recall)   marqo/fashionSigLIP encode_image(ref_img)
                       → supabase.rpc("match_clothes") → top-200 후보
  ② Stage 2 (Re-rank)  후보 200장 비동기 다운로드
                       → BLIP-2 ViT-G (4-bit nf4 우선 / bf16 fallback)
                       → composed query: Q-Former(ref_img, text=final_text)
                       → candidate    : Q-Former(cand_img)
                       → multi-head matching (token mix 32→12, channel mix 768→256, cosine 합산)
                       → top-20 반환

기존 main.py(8000), gemini_search_main.py(8001) 와 독립.
DB(Supabase)의 임베딩은 marqo/fashionSigLIP 으로 만들어졌으므로 Stage 1 모델도 동일 모델 사용 필수.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import traceback

import httpx
import open_clip
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from supabase import Client, create_client
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

# Gemini 후처리 (gemini_search_main.py 와 동일 패턴 — 모델 중복 로드 회피 위해 import 대신 재정의)
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


# ──────────────────────────────────────────────────────────────
# 1. 환경 변수 & 외부 클라이언트
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

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
    print("⚠️  GEMINI_API_KEY 없음 / google-genai 미설치 → Gemini 후처리 비활성, rule-based fallback 사용")


# ──────────────────────────────────────────────────────────────
# 2. 모델 로딩
# ──────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device = {device}")

# (a) fashionSigLIP — Stage 1 image embedding
FS_MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
print(f"[init] loading fashionSigLIP: {FS_MODEL_ID}")
fs_model, _, fs_preprocess = open_clip.create_model_and_transforms(FS_MODEL_ID)
fs_model = fs_model.to(device)
fs_model.eval()
print("[init] fashionSigLIP loaded")

# (b) BLIP-2 ViT-G — Stage 2 cross-encoder. 4-bit nf4 우선, 실패 시 bf16 fallback.
BLIP2_ID = "Salesforce/blip2-itm-vit-g"
print(f"[init] loading BLIP-2: {BLIP2_ID}")
blip2_proc = AutoProcessor.from_pretrained(BLIP2_ID)

QUANT_MODE = "bf16"
COMPUTE_DTYPE = torch.bfloat16
blip2 = None
try:
    # bitsandbytes 가 import 가능하고 GPU 가 있을 때만 nf4 시도
    if device == "cuda":
        import bitsandbytes  # noqa: F401 — import 가능성 체크
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        blip2 = Blip2ForImageTextRetrieval.from_pretrained(
            BLIP2_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        QUANT_MODE = "nf4"
        print("[init] BLIP-2 loaded with 4-bit nf4")
except Exception as e:
    print(f"[init] nf4 로드 실패 → bf16 fallback: {e}")
    blip2 = None  # 재시도 위해 명시적 초기화

if blip2 is None:
    # bf16 fallback (또는 CPU)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    COMPUTE_DTYPE = dtype
    blip2 = Blip2ForImageTextRetrieval.from_pretrained(BLIP2_ID, dtype=dtype)
    blip2 = blip2.to(device)
    QUANT_MODE = "bf16" if dtype == torch.bfloat16 else "fp32"
    print(f"[init] BLIP-2 loaded with {QUANT_MODE}")

blip2.eval()
NUM_QUERY = int(blip2.config.num_query_tokens)  # 보통 32
print(f"[init] num_query_tokens = {NUM_QUERY}")

# transformers 5.x: blip2-itm-vit-g 의 AutoProcessor 가 num_query_tokens 를 None 으로 두고 로드되어
# 텍스트와 함께 호출할 때 `max_length - self.num_query_tokens` 에서 TypeError 발생.
# model.config 에서 가져와 명시적으로 주입.
if getattr(blip2_proc, "num_query_tokens", None) is None:
    try:
        blip2_proc.num_query_tokens = NUM_QUERY
        print(f"[init] patched blip2_proc.num_query_tokens = {NUM_QUERY}")
    except Exception as _e:
        print(f"[init] num_query_tokens patch 실패 (무시): {_e}")


# ──────────────────────────────────────────────────────────────
# 3. Pydantic 스키마 (Gemini Vision 응답)
#    gemini_search_main.py 와 동일 정의 — Schema 형태는 그대로.
# ──────────────────────────────────────────────────────────────
class ReferenceImageAttributes(BaseModel):
    category: str = Field(description="garment type, e.g. hoodie, jeans")
    color: str = Field(description="dominant color in plain English")
    fit: str = Field(description="silhouette / fit")
    pattern: str = Field(description="solid, striped, graphic print, etc.")
    material: str = Field(description="fabric if visible")
    details: str = Field(description="distinctive details")


def _empty_attrs() -> ReferenceImageAttributes:
    return ReferenceImageAttributes(category="", color="", fit="", pattern="", material="", details="")


# ──────────────────────────────────────────────────────────────
# 4. Gemini 후처리 (gemini_search_main.py 의 동일 함수 복제)
# ──────────────────────────────────────────────────────────────
VISION_SYSTEM_PROMPT = """
You are a fashion attribute extractor. Look at the single garment in the photo
and extract its attributes as JSON. Use concise English noun phrases. If a
field cannot be determined from the image, return an empty string for that
field. Do NOT invent attributes that are not visible.
""".strip()

FUSION_SYSTEM_PROMPT = """
You are a fashion search prompt composer.
Inputs:
  - REFERENCE_ATTRS: a JSON object describing a reference garment.
  - USER_QUERY: a user's modification request (Korean or English).

Task:
  Produce ONE concise English description of the target garment the user is
  looking for. If USER_QUERY explicitly changes an attribute, OVERRIDE that
  attribute and keep the rest from REFERENCE_ATTRS. Output the description
  only — no quotes, no prefix, no JSON. Keep under 30 words.
""".strip()


def _guess_mime(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            fmt = (im.format or "").lower()
        if fmt == "png":
            return "image/png"
        if fmt == "webp":
            return "image/webp"
        return "image/jpeg"
    except Exception:
        return "image/jpeg"


def analyze_reference_image(image_bytes: bytes) -> ReferenceImageAttributes:
    if gemini_client is None or genai_types is None:
        return _empty_attrs()
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                VISION_SYSTEM_PROMPT,
                genai_types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime(image_bytes)),
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReferenceImageAttributes,
            ),
        )
        parsed = getattr(resp, "parsed", None)
        data = parsed.model_dump() if (parsed is not None and hasattr(parsed, "model_dump")) else json.loads(resp.text)
        return ReferenceImageAttributes(
            category=str(data.get("category", "") or ""),
            color=str(data.get("color", "") or ""),
            fit=str(data.get("fit", "") or ""),
            pattern=str(data.get("pattern", "") or ""),
            material=str(data.get("material", "") or ""),
            details=str(data.get("details", "") or ""),
        )
    except Exception as e:
        print(f"[gemini-vision] failed → empty attrs: {e}")
        return _empty_attrs()


def _rule_based_fusion(attrs: ReferenceImageAttributes, user_query: str) -> str:
    parts = [attrs.fit, attrs.color, attrs.category]
    base = " ".join(p for p in parts if p).strip()
    extras = ", ".join(p for p in [attrs.pattern, attrs.material, attrs.details] if p)
    description = base if not extras else f"{base}, {extras}"
    uq = user_query.strip()
    if not description:
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
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[FUSION_SYSTEM_PROMPT, prompt],
        )
        text = (getattr(resp, "text", "") or "").strip()
        return text if text else _rule_based_fusion(attrs, user_query)
    except Exception as e:
        print(f"[gemini-text] failed → rule-based: {e}")
        return _rule_based_fusion(attrs, user_query)


# ──────────────────────────────────────────────────────────────
# 5. Stage 1 — fashionSigLIP image-only recall
#    main.py:251-280 패턴
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def stage1_recall(ref_pil: Image.Image, top_n: int) -> list[dict]:
    tensor = fs_preprocess(ref_pil).unsqueeze(0).to(device)
    feats = fs_model.encode_image(tensor, normalize=True)
    embedding = feats.squeeze().detach().cpu().tolist()
    resp = supabase.rpc(
        "match_clothes",
        {
            "query_embedding": embedding,
            "match_threshold": 0.0,  # 임계값 없이 모두 후보로
            "match_count": top_n,
            "filter_main_category": None,
            "filter_sub_category": None,
        },
    ).execute()
    return resp.data or []


# ──────────────────────────────────────────────────────────────
# 6. 후보 이미지 비동기 다운로드
# ──────────────────────────────────────────────────────────────
async def fetch_candidate_images(items: list[dict]) -> list[tuple[dict, Image.Image | None]]:
    """200개 안팎의 candidate 이미지를 병렬 다운로드. 실패한 항목은 (item, None)."""
    timeout = httpx.Timeout(connect=4.0, read=6.0, write=4.0, pool=6.0)
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        async def one(it: dict) -> tuple[dict, Image.Image | None]:
            url = (it.get("image_url") or "").strip()
            if not url:
                return (it, None)
            if url.startswith("//"):
                url = "https:" + url
            try:
                r = await client.get(url)
                r.raise_for_status()
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
                return (it, im)
            except Exception as e:
                print(f"[stage2] download fail {url[:80]}…: {e}")
                return (it, None)

        return await asyncio.gather(*(one(it) for it in items))


# ──────────────────────────────────────────────────────────────
# 7. Stage 2 — Multi-head matching (FashionBLIP-2 inspired)
#
#    FACap 논문 핵심:
#      - Q-Former 32 query token → token-mix 12 heads → channel-mix 256 dim
#      - 헤드별 cosine similarity 합산이 최종 점수
#    학습된 token/channel 어댑터가 없으므로 deterministic 근사:
#      - Token mix : 32 토큰을 12 그룹으로 균등 분할 → 그룹 평균
#      - Channel mix: 앞 256 채널 truncate
# ──────────────────────────────────────────────────────────────
NT, DC = 12, 256


def _token_mix_groups(nq: int, nt: int = NT) -> list[list[int]]:
    """32 토큰을 12 그룹으로 균등 분할 (크기 차이 ≤ 1)."""
    base, rem = divmod(nq, nt)
    sizes = [base + 1] * rem + [base] * (nt - rem)
    groups, i = [], 0
    for s in sizes:
        groups.append(list(range(i, i + s)))
        i += s
    return groups


GROUPS = _token_mix_groups(NUM_QUERY)


def multi_head_reduce(feats: torch.Tensor) -> torch.Tensor:
    """(num_query, hidden) → (NT, DC) deterministic 근사."""
    # Token mix: 그룹 평균
    pooled = torch.stack([feats[g].mean(dim=0) for g in GROUPS], dim=0)  # (NT, hidden)
    # Channel mix: 앞 DC 채널 truncate (hidden < DC 인 경우 zero-pad)
    if pooled.shape[-1] >= DC:
        return pooled[:, :DC]
    pad = torch.zeros(pooled.shape[0], DC - pooled.shape[-1], device=pooled.device, dtype=pooled.dtype)
    return torch.cat([pooled, pad], dim=-1)


def multi_head_score(q: torch.Tensor, c: torch.Tensor) -> float:
    """q, c: (NT, DC) → 헤드별 cosine 합산 (총 NT 헤드)."""
    qn = F.normalize(q.to(torch.float32), dim=-1)
    cn = F.normalize(c.to(torch.float32), dim=-1)
    return (qn * cn).sum(dim=-1).sum().item()


@torch.no_grad()
def _qformer_image_only(pil: Image.Image) -> torch.Tensor:
    """이미지만 받아 Q-Former (32, hidden) 출력."""
    inputs = blip2_proc(images=pil, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device, dtype=COMPUTE_DTYPE)

    vision_out = blip2.vision_model(pixel_values=pixel_values)
    image_embeds = vision_out[0] if isinstance(vision_out, tuple) else vision_out.last_hidden_state
    image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    qt = blip2.query_tokens.expand(image_embeds.shape[0], -1, -1).to(image_embeds.dtype)
    qout = blip2.qformer(
        query_embeds=qt,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_mask,
    )
    out = qout[0] if isinstance(qout, tuple) else qout.last_hidden_state  # (B, NUM_QUERY, hidden)
    return out[0]  # (NUM_QUERY, hidden)


@torch.no_grad()
def _qformer_image_text(pil: Image.Image, text: str) -> torch.Tensor:
    """이미지 + 텍스트 attended Q-Former → 처음 NUM_QUERY 토큰 (32, hidden).

    주의: blip2_proc(images=..., text=...) 를 한 번에 호출하면 input_ids 앞에 image_token id 가
    NUM_QUERY 개 삽입되어 Q-Former 의 BERT vocab 범위(30522)를 벗어남 → IndexError.
    image processor 와 tokenizer 를 분리해서 호출하면 텍스트 input_ids 가 vocab 내로 유지됨.
    Q-Former 의 query 자리는 이후 self.embeddings(input_ids, query_embeds) 호출 시 자동으로 결합됨.
    """
    img_inputs = blip2_proc(images=pil, return_tensors="pt")
    pixel_values = img_inputs.pixel_values.to(device, dtype=COMPUTE_DTYPE)

    txt_inputs = blip2_proc.tokenizer(
        text or " ",
        return_tensors="pt",
        truncation=True,
        max_length=32,  # query 32 + text 32 = 총 64 (Q-Former max_position_embeddings 와 일치)
        padding=True,
    )
    input_ids = txt_inputs.input_ids.to(device)
    attn_mask = txt_inputs.attention_mask.to(device)

    vision_out = blip2.vision_model(pixel_values=pixel_values)
    image_embeds = vision_out[0] if isinstance(vision_out, tuple) else vision_out.last_hidden_state
    image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    qt = blip2.query_tokens.expand(image_embeds.shape[0], -1, -1).to(image_embeds.dtype)
    query_attn = torch.ones((qt.shape[0], qt.shape[1]), dtype=torch.long, device=qt.device)
    full_mask = torch.cat([query_attn, attn_mask], dim=1)

    # embeddings() 가 (query_tokens, text_embeddings) 를 결합하여 (B, num_query+T, hidden) 반환
    qe = blip2.embeddings(input_ids=input_ids, query_embeds=qt)

    qout = blip2.qformer(
        query_embeds=qe,
        query_length=qt.shape[1],
        attention_mask=full_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_mask,
    )
    out = qout[0] if isinstance(qout, tuple) else qout.last_hidden_state
    return out[0, : qt.shape[1], :]  # (NUM_QUERY, hidden)


# ──────────────────────────────────────────────────────────────
# 8. FastAPI
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="TexTyle Precision Search Server")

STAGE1_COUNT = int(os.environ.get("STAGE1_COUNT", "50"))
STAGE2_TOP = int(os.environ.get("STAGE2_TOP", "10"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "quant_mode": QUANT_MODE,
        "gemini_enabled": gemini_client is not None,
        "stage1_count": STAGE1_COUNT,
        "stage2_top": STAGE2_TOP,
        "num_query_tokens": NUM_QUERY,
        "heads": NT,
        "head_dim": DC,
    }


@app.post("/precision_search")
async def precision_search(
    file: UploadFile = File(None),
    query: str = Form(None),
):
    if not file or query is None or not query.strip():
        raise HTTPException(status_code=400, detail="이미지(file)와 요청 텍스트(query) 둘 다 필요합니다.")

    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="이미지 파일이 비어있습니다.")
        ref_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ⓪ Gemini 후처리
        attrs = analyze_reference_image(img_bytes)
        final_text = fuse_with_user_query(attrs, query)
        print(f"[precision] final_text = {final_text!r}")

        # ① Stage 1: image-only recall (top-200)
        candidates = stage1_recall(ref_pil, top_n=STAGE1_COUNT)
        if not candidates:
            return {
                "message": "Empty stage1",
                "quant_mode": QUANT_MODE,
                "gemini_enabled": gemini_client is not None,
                "image_attributes": attrs.model_dump(),
                "final_text": final_text,
                "stage1_count": 0,
                "stage2_evaluated": 0,
                "results": [],
            }

        # 후보 이미지 비동기 다운로드
        paired = await fetch_candidate_images(candidates)
        paired = [(it, im) for it, im in paired if im is not None]

        # ② Stage 2: BLIP-2 multi-head re-rank
        # composed query: ref_img + final_text (text-attended Q-Former)
        q_feats = multi_head_reduce(_qformer_image_text(ref_pil, final_text))

        scored: list[dict] = []
        for it, cim in paired:
            try:
                c_feats = multi_head_reduce(_qformer_image_only(cim))
                s = multi_head_score(q_feats, c_feats)
            except Exception as e:
                print(f"[stage2] qformer fail (id={it.get('id')}): {e}")
                s = float("-inf")
            entry = dict(it)
            entry["mh_score"] = s
            entry.pop("embedding", None)  # 페이로드 절약
            scored.append(entry)

        scored.sort(key=lambda x: x["mh_score"], reverse=True)
        top = scored[:STAGE2_TOP]

        return {
            "message": "Success",
            "quant_mode": QUANT_MODE,
            "gemini_enabled": gemini_client is not None,
            "image_attributes": attrs.model_dump(),
            "final_text": final_text,
            "stage1_count": len(candidates),
            "stage2_evaluated": len(paired),
            "results": top,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────
# 9. 실행부
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
