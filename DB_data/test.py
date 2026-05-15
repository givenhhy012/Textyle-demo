import os
import sys
import torch
import requests
import time
from PIL import Image
from io import BytesIO
from supabase import create_client, Client
from dotenv import load_dotenv
import re

# 🔥 에러를 일으키는 transformers(AutoModel, AutoProcessor) 대신 open_clip을 직접 사용합니다!
import open_clip

# -------------------------------------------------------------
# 1. 환경 변수 및 Supabase 연결 설정
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ .env 파일에서 Supabase 정보를 불러오지 못했습니다.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------------------
# 2. AI 모델 로드 (open_clip 직접 사용으로 버그 완벽 우회!)
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# open_clip 전용 모델 주소 (앞에 hf-hub: 가 붙습니다)
model_id = "hf-hub:Marqo/marqo-fashionSigLIP"

print("⏳ AI 모델 로딩 중... (약간의 시간이 소요됩니다)")
# 모델과 전처리 도구(preprocess)를 오류 없이 안전하게 바로 가져옵니다.
model, _, preprocess = open_clip.create_model_and_transforms(model_id)
model = model.to(device)
print("✅ AI 모델 로딩 완료!")

# -------------------------------------------------------------
# 3. 카테고리 맵핑 설정
# -------------------------------------------------------------
CATEGORY_MAP = {
    # 아우터 (002) 17종류
    "002020": "가디건", "002008": "환절기 코트", "002023": "플리스", 
    "002027": "경량패딩/패딩 베스트", "002007": "겨울 싱글코트", "002025": "무스탕", 
    "002009": "겨울 기타코트", "002013": "롱패딩", "002012": "숏패딩", 
    "002024": "겨울 더블코트", "002022": "후드집업", "002017": "트러커자켓", 
    "002001": "블루종/MA-1", "002006": "코치자켓", "002002": "레더자켓", 
    "002003": "슈트/블레이저 자켓", "002014": "사파리/헌팅자켓",
    
    # 상의 (001) 7종류
    "001010": "긴소매 티셔츠", "001005": "스웻셔츠", "001002": "셔츠", 
    "001001": "반소매 티셔츠", "001006": "니트/스웨터", "001003": "피케/카라 티셔츠", 
    "001004": "후드티",
    
    # 하의 (003) 5종류
    "003002": "데님팬츠", "003004": "트레이닝/조거 팬츠", "003007": "코튼 팬츠", 
    "003008": "슬랙스/슈트 팬츠", "003009": "숏팬츠"
}

def get_categories_from_code(category_code: str):
    sub_category = CATEGORY_MAP.get(category_code, "기타")
    prefix = category_code[:3]
    if prefix == "001": main_category = "상의"
    elif prefix == "002": main_category = "아우터"
    elif prefix == "003": main_category = "하의"
    else: main_category = "기타"
    return main_category, sub_category

def to_high_res_url(thumbnail_url: str) -> str:
    """무신사 썸네일 URL(_500.jpg)을 상세페이지 고화질 URL(_big.jpg)로 변환."""
    if not thumbnail_url:
        return thumbnail_url
    high_res = thumbnail_url.replace('/images/goods_img/', '/thumbnails/images/goods_img/')
    high_res = high_res.replace('_500.jpg', '_big.jpg')
    return high_res

# -------------------------------------------------------------
# 4. DB 저장 및 AI 임베딩 함수
# -------------------------------------------------------------
def insert_clothes_data(name: str, image_url: str, shop_link: str, main_category: str, sub_category: str, price: int, brand_name: str):
    print(f"🔄 처리 중: [{brand_name}] [{main_category} > {sub_category}] {name} - {price}원")
    
    try:
        existing_data = supabase.table("clothes").select("image_url").eq("image_url", image_url).execute()
        
        if len(existing_data.data) > 0:
            print(f"⏩ 이미 등록된 상품입니다. 처리를 건너뜁니다: {name}")
            return  

        # 1. 고화질 이미지 다운로드 (썸네일 → _big.jpg 변환)
        high_res_url = to_high_res_url(image_url)
        print(f"  -> 새 상품 확인됨. 고화질 이미지 다운로드 시작: {high_res_url}")
        try:
            response = requests.get(high_res_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # 고화질 URL이 없는 상품은 썸네일로 폴백
            print(f"  ⚠️ 고화질 이미지 실패({e}), 썸네일로 폴백합니다.")
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 2. open_clip 방식으로 이미지 전처리 및 벡터 변환
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor, normalize=True)
            embedding_list = image_features.squeeze().tolist()

        # 3. Supabase 삽입
        data, count = supabase.table("clothes").upsert({
            "brand_name": brand_name,
            "name": name,
            "main_category": main_category,   
            "sub_category": sub_category,     
            "price": price,
            "image_url": image_url,
            "shop_link": shop_link,
            "embedding": embedding_list
        }).execute()

        print(f"✅ 성공적으로 DB에 저장되었습니다: [{brand_name}] {name}")

    except Exception as e:
        print(f"❌ 데이터 처리/삽입 실패: {e}")

# -------------------------------------------------------------
# 5. 무신사 크롤링 함수
# -------------------------------------------------------------
def crawl_musinsa_and_save(api_url, category_code):
    print("카테고리 코드:",category_code)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.musinsa.com/"
    }

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status() 
        
        json_data = response.json()
        item_list = json_data.get('data', {}).get('list', [])
        
        if not item_list:
            print("⚠️ 가져올 상품 데이터가 없습니다.")
            return

        print(f"📦 총 {len(item_list)}개의 상품 데이터를 찾았습니다. DB 저장을 시작합니다.")

        for item in item_list:
            name = item.get('goodsName', '이름없음')
            price = item.get('price', 0)
            brand = item.get('brandName', '브랜드없음')
            
            img_url = item.get('thumbnail', '')
            if img_url and img_url.startswith('//'):
                img_url = 'https:' + img_url
                
            shop_link = item.get('goodsLinkUrl', '')
            main_cat, sub_cat = get_categories_from_code(category_code)
            
            insert_clothes_data(name, img_url, shop_link, main_cat, sub_cat, price, brand)
            time.sleep(0.5)

    except Exception as e:
        print(f"❌ 크롤링 중 에러 발생: {e}")

if __name__ == "__main__":
    api_url = "https://api.musinsa.com/api2/dp/v2/plp/goods?gf=M&sortCode=POPULAR&category=002022&size=60&testGroup=&caller=CATEGORY&page=5&seen=240&seenAds=&hmacId=7776f518dd934633cd93bd14318bf8a56d53655a8da4fa5361d6bf536b8eae80"
    category_code = re.search(r'category=(\d+)', api_url)
    crawl_musinsa_and_save(api_url, category_code.group(1))