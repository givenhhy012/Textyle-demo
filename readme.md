## 📁 프로젝트 구조

```
Textyle-demo/
├── Textyle-app/                    # React Native 모바일 앱 (Expo)
│   ├── app/(tabs)/
│   │   ├── index.tsx               # 기본 검색 (포트 8000)
│   │   ├── ai_search.tsx           # AI 검색 (Gemini + fashionSigLIP, 포트 8001)
│   │   ├── precision_search.tsx    # AI 정밀검색 (BLIP-2 re-rank, 포트 8002) [신규]
│   │   └── ...
│   └── app.json
│
├── Textyle-vectorserver/           # FastAPI 백엔드 (3개 포트)
│   ├── main.py                     # 포트 8000 — 기본 텍스트/이미지 벡터 검색
│   ├── gemini_search_main.py       # 포트 8001 — Gemini Vision/Text 융합 + fashionSigLIP
│   ├── precision_search_main.py    # 포트 8002 — Gemini 전처리 + fashionSigLIP recall + BLIP-2 re-rank [신규]
│   ├── .env                        # API 키 (Supabase, Gemini, OpenAI 등)
│   ├── colab_precision_search.ipynb # Google Colab 에서 실행할 노트북 [신규]
│   └── COLAB_README.md             # Colab 상세 가이드 [신규]
│
└── DB_data/                        # 데이터베이스 관련 (Supabase)
```

---

## 🚀 [로컬 환경] 전체 실행 가이드

### 1️⃣ 사전 준비 (한 번만)

(참고사항: 데이터 적재는 test.py 이용했음)

#### 1.1 Python 의존성 설치
```powershell
cd Textyle-vectorserver
pip install -r requirements.txt
# 또는 수동 설치:
pip install fastapi uvicorn python-multipart pillow supabase python-dotenv open_clip_torch torch transformers google-genai bitsandbytes accelerate httpx
```

#### 1.2 Node.js 의존성 설치
```powershell
cd Textyle-app
npm install
# 또는 yarn install
```

#### 1.3 환경 변수 설정
**`Textyle-vectorserver/.env` 파일 생성 (공유받은 값 붙여넣기):**
```env
SUPABASE_URL= {붙여넣기}
SUPABASE_KEY= {붙여넣기}
GEMINI_API_KEY= {gemini ai studio에서 키 발급받아 사용}
GEMINI_MODEL_NAME=gemini-3.1-flash-lite

# 선택사항 (기본값이 있음):
STAGE1_COUNT=200
STAGE2_TOP=20
AI_SEARCH_THRESHOLD=0.15
```

**`Textyle-app/.env` 파일 생성:**
```env
EXPO_PUBLIC_SUPABASE_URL= {값은 벡터서버와 동일}
EXPO_PUBLIC_SUPABASE_ANON_KEY=

GEMINI_API_KEY=
GEMINI_MODEL_NAME=gemini-3.1-flash-lite

```

#### 1.4 IP 주소 확인 및 변경
**로컬 PC의 IP 주소 확인 (PowerShell):**
```powershell
ipconfig
# IPv4 주소 확인: 192.168.x.x 형태 (또는 다른 형태)
```

**앱 코드에서 IP 변경:**
- `Textyle-app/app/(tabs)/index.tsx` 줄 60-61
- `Textyle-app/app/(tabs)/ai_search.tsx` 줄 59
- `Textyle-app/app/(tabs)/precision_search.tsx` 줄 62

```tsx
const SERVER_BASE_URL = "http://192.168.0.40:8002";  // 본인 IP로 변경
```

---

### 2️⃣ 서버 실행 (터미널 2개 필요)

서버를 각각 따로 실행해서 테스트해보는 것 권장(노트북 환경에서의 성능 문제). (앱 터미널 1개, 서버 터미널 1개)

**[1] 벡터 서버 - main.py (포트 8000)**
=> marqo/fashionSigLIP 모델 이용한 단순 벡터 합.
```powershell
cd Textyle-vectorserver
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
✅ 성공 시: `INFO:     Uvicorn running on http://0.0.0.0:8000`

**[2] Gemini 검색 서버 - gemini_search_main.py (포트 8001)**
=> 입력 이미지를 Gemini가 해석, 장문의 텍스트로 변환. 입력 텍스트를 Gemini가 장문으로 확장. 이후 두 텍스트를 융합하여 하나의 장문텍스트를 marqo/fashionSigLIP모델에서 검색.
```powershell
cd Textyle-vectorserver
uvicorn gemini_search_main:app --host 0.0.0.0 --port 8001 --reload
```
✅ 성공 시: `INFO:     Uvicorn running on http://0.0.0.0:8001`

**[3] 정밀 검색 서버 - precision_search_main.py (포트 8002)** (선택, 정밀검색 탭만 사용 시)
=> 입력 이미지를 이용, marqo/fashionShiLIP모델로 비슷한 옷 후보 추림. gemini 이용하여 요구사항 텍스트를 정제. FashionBLIP-2 이용하여 (ref img, text, top-200)셋을 정밀 검사, 최종 결과 선정.(Google Colab 이용해야함.)

```powershell
cd Textyle-vectorserver
uvicorn precision_search_main:app --host 0.0.0.0 --port 8002 --reload
```
✅ 성공 시: `INFO:     Uvicorn running on http://0.0.0.0:8002`

---

### 3️⃣ 앱 실행

현재 DB에 아우터>후드집업 카테고리의 데이터만 저장되어있으니 검색 시 참고.

**[터미널 4] 모바일 앱**
```powershell
cd Textyle-app
npx expo start
# 또는 더 자세한 로그:
npx expo start --verbose
```

스마트폰에서:
- iOS: Expo Go 앱 → QR 코드 스캔
- Android: Expo Go 앱 → QR 코드 스캔

---


## ☁️ [Google Colab] 클라우드 실행 가이드(정밀 검색 서버 - precision_search_main.py 이용시)

로컬 GPU가 부족하면 Colab 의 무료 T4 GPU 사용 권장. 응답 시간은 ~15-20초 (로컬 대비 3-4배).

### 1️⃣ Colab 노트북 준비

1. Google Colab 열기: https://colab.research.google.com
2. `파일 → 노트북 업로드`
3. `Textyle-vectorserver/colab_precision_search.ipynb` 선택
4. **런타임 → 런타임 유형 변경 → GPU (T4)**

### 2️⃣ 필수 준비물

- ngrok 무료 계정 + 토큰: https://dashboard.ngrok.com/get-started/your-authtoken (2분)
- Colab Secrets (🔑 아이콘) 등록:
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
  - `GEMINI_API_KEY`
  - `NGROK_AUTH_TOKEN`

### 3️⃣ 노트북 실행 순서

| 셀 | 내용 | 소요시간 |
|---|---|---|
| 1 | GPU 확인 (`nvidia-smi`) | 10초 |
| 2 | 의존성 설치 | ~3분 |
| 3 | 환경변수 로드 | 10초 |
| 4 | 파일 체크 | 5초 |
| 5 | ngrok 터널 생성 → **URL 복사** 📌 | 10초 |
| 6 | 서버 실행 + 헬스체크 | 5분 (첫실행 모델 다운로드) |

### 4️⃣ Colab 서버 주소를 앱에 적용

셀 5에서 출력된 ngrok URL (예: `https://abcd1234.ngrok-free.app`) 을 복사하여

**`Textyle-app/app/(tabs)/precision_search.tsx` 줄 62 수정:**
```tsx
const SERVER_BASE_URL = "https://abcd1234.ngrok-free.app";  // https, 포트 없음
```

저장하면 Expo 가 자동 hot-reload.

### 5️⃣ Colab 트러블슈팅

더 자세한 내용은 **[Textyle-vectorserver/COLAB_README.md](Textyle-vectorserver/COLAB_README.md)** 참고.

---

## 📱 앱 테스트 (각 탭 확인)

앱 실행 후 Expo GO 에서 다음을 차례로 테스트하세요.

### 탭 1️⃣ — 검색 (포트 8000)

### 탭 2️⃣ — AI 검색 (포트 8001)

### 탭 3️⃣ — AI 정밀검색 (포트 8002, 선택)

---




## 📚 주요 파일 참고

| 파일 | 설명 |
|---|---|
| [Textyle-vectorserver/precision_search_main.py](Textyle-vectorserver/precision_search_main.py) | 정밀검색 서버 코드 |
| [Textyle-app/app/(tabs)/precision_search.tsx](Textyle-app/app/(tabs)/precision_search.tsx) | 정밀검색 앱 화면 |
| [Textyle-vectorserver/COLAB_README.md](Textyle-vectorserver/COLAB_README.md) | Colab 전체 가이드 |
