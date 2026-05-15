# 🎯 Google Colab 에서 Precision Search 서버 실행하기

`precision_search_main.py` 는 **BLIP-2 ViT-G** (~14GB) 를 사용해 무거우므로,  
로컬 노트북 GPU가 부족한 경우 **Google Colab T4 무료 GPU** 에서 돌리고 ngrok 으로 모바일 앱과 연결합니다.

---

## 📋 사전 준비

| 항목 | 어디서 |
|---|---|
| Google 계정 | https://colab.research.google.com |
| Colab GPU 런타임 | 메뉴: **런타임 → 런타임 유형 변경 → T4 GPU** |
| Supabase URL / KEY | 기존 `.env` 에 있는 값 그대로 |
| Gemini API KEY | 기존 `.env` 에 있는 값 그대로 |
| **ngrok 무료 토큰** | https://dashboard.ngrok.com/get-started/your-authtoken (가입 5초) |

---

## 🚀 실행 절차

### 1. Colab 노트북 열기
[colab_precision_search.ipynb](colab_precision_search.ipynb) 파일을 Colab 에 업로드 (`파일 → 노트북 업로드`).

### 2. GPU 런타임 활성화
런타임 유형을 **T4 GPU** 로 변경. (Pro 계정이면 V100/L4 권장)

### 3. Secrets 등록 (권장)
Colab 좌측 🔑 자물쇠 아이콘 클릭 → 4개 추가하고 모두 **Notebook access ON**:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `GEMINI_API_KEY`
- `NGROK_AUTH_TOKEN`

### 4. `precision_search_main.py` 업로드
좌측 파일 패널에 드래그하여 업로드. (`.env` 는 필요 없음 — 환경변수로 대체)

### 5. 셀 순서대로 실행
1. **셀 1** `nvidia-smi` — T4 인지 확인
2. **셀 2** 의존성 설치 (~3분)
3. **셀 3** 환경변수 로드
4. **셀 4** 파일 점검
5. **셀 5** ngrok 터널 생성 → `https://xxxxx.ngrok-free.app` 발급
6. **셀 6** 서버 실행 + 헬스체크 (초기 모델 다운로드 ~5분, 이후 ~30초)

### 6. 앱에 URL 적용
셀 5에서 출력된 ngrok URL 을 복사하여 앱 코드에 입력:

[Textyle-app/app/(tabs)/precision_search.tsx](../Textyle-app/app/%28tabs%29/precision_search.tsx) 의 `SERVER_BASE_URL` 변경:

```tsx
// 로컬 PC 서버 사용 시
const SERVER_BASE_URL = "http://192.168.0.40:8002";

// Colab + ngrok 사용 시 (← 여기로 교체)
const SERVER_BASE_URL = "https://xxxxx.ngrok-free.app";
```

저장 후 Expo 가 자동 hot-reload 합니다.

---

## ⚠️ 주의사항

### ngrok 무료 한계
- **요청 헤더 경고 페이지**: 무료 도메인은 처음 호출 시 브라우저 경고를 보일 수 있음. RN fetch 는 영향 없음.
- **세션 유지**: ngrok 무료 토큰은 1개 터널만 가능. 같은 노트북 여러 번 셀5 실행 시 새 URL 발급됨.
- **URL 매번 바뀜**: 매 실행마다 새 도메인. 앱 코드 매번 수정해야 함 (또는 고정 도메인 유료 플랜).

### Colab 무료 한계
- **연속 12시간 / 일일 사용량 제한**: 장기 서버 운영용 X. 테스트/개발용 ✓
- **유휴 90분 자동 종료**: 셀 실행 안 하면 끊김. 가끔 셀6 다시 누르거나 keep-alive 확장 사용.
- **GPU 미보장**: 트래픽 많을 때 CPU 만 할당될 수 있음. 그 경우 BLIP-2 가 bf16 fallback 으로 매우 느려짐.

### 성능
| 환경 | Stage 1 | Stage 2 | 총 응답 |
|---|---|---|---|
| 로컬 RTX 4090 (bf16) | ~1초 | ~3-5초 | **~5초** |
| Colab T4 (nf4) | ~2초 | ~10-15초 | **~15-20초** |
| Colab CPU (bf16 fallback) | ~5초 | ~3-5분 | 권장 X |

### 모델 캐시
첫 실행 시 BLIP-2 ViT-G (~14GB) 다운로드. 같은 세션 내에서는 캐시되지만, **세션 끊기면 다시 다운로드**.  
세션 유지하려면 셀6 한번만 실행하고 그대로 두기.

---

## 🐛 트러블슈팅

### "❌ NGROK_AUTH_TOKEN 미설정"
→ Colab Secrets 에 `NGROK_AUTH_TOKEN` 추가 후 셀3 재실행.

### "❌ precision_search_main.py 가 없습니다"
→ 좌측 파일 패널에 .py 파일 드래그 업로드.

### "RuntimeError: CUDA out of memory"
→ 다른 Colab 노트북 실행 중인지 확인. `런타임 → 런타임 재시작`.

### "ImportError: bitsandbytes"
→ Colab T4 가 아닌 환경. 코드가 자동으로 bf16 fallback 으로 빠지므로 동작은 되지만 느려짐.

### 앱에서 "통신 에러" 발생
→ ngrok URL 정확히 복사했는지 확인. **`https://` 포함, 끝에 `/` 없이.**  
→ Colab 셀6 출력에 `✅ 서버 정상!` 떴는지 확인.

### "Cannot read response.json() — html"
→ ngrok 무료 도메인의 브라우저 경고 페이지일 수 있음.  
→ 노트북 셀에서 한번 `requests.get(public_url)` 호출해서 경고 통과시키거나, ngrok 유료 도메인 사용.

---

## 🔄 로컬 ↔ Colab 전환

`precision_search.tsx` 상단의 `SERVER_BASE_URL` 한 줄만 바꾸면 됩니다.  
원한다면 두 값을 상수로 정의하고 토글:

```tsx
const SERVER_BASE_URL = __DEV__
  ? "http://192.168.0.40:8002"          // 로컬 PC
  : "https://xxxxx.ngrok-free.app";     // Colab
```
