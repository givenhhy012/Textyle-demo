import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import { ActivityIndicator, Alert, Image, Linking, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

/**
 * AI 정밀검색 화면.
 * 흐름: ⓪ Gemini 후처리 → ① fashionSigLIP recall (top-200) → ② BLIP-2 multi-head re-rank (top-20)
 * 백엔드: Textyle-vectorserver/precision_search_main.py (port 8002) 의 POST /precision_search
 */
export default function PrecisionSearchScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [searchText, setSearchText] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [hasSearched, setHasSearched] = useState(false);

  // 서버 디버그/상태 정보
  const [finalText, setFinalText] = useState<string>('');
  const [stageInfo, setStageInfo] = useState<{ stage1?: number; stage2?: number; quant?: string }>({});

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 5],
      quality: 0.8,
    });
    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  };

  const searchClothes = async () => {
    if (!imageUri || !searchText.trim()) {
      Alert.alert('알림', '사진과 요청사항을 모두 입력해주세요!');
      return;
    }

    setIsLoading(true);
    setFinalText('');
    setStageInfo({});

    try {
      const formData = new FormData();
      const uriParts = imageUri.split('.');
      const fileType = uriParts[uriParts.length - 1];

      formData.append('file', {
        uri: imageUri,
        name: `photo.${fileType}`,
        type: `image/${fileType}`,
      } as any);

      formData.append('query', searchText.trim());

      // 🚨 서버 base URL 설정 — 둘 중 한 줄만 활성화하세요.
      //   (A) 로컬 PC 서버    : http://<PC_IP>:8002
      //   (B) Colab + ngrok   : https://xxxxx.ngrok-free.app  (포트 없음, https 필수)
      // Colab 사용법은 Textyle-vectorserver/COLAB_README.md 참고.
      const SERVER_BASE_URL = "https://rocker-proactive-exalted.ngrok-free.dev";
      // const SERVER_BASE_URL = "https://xxxxx.ngrok-free.app";

      const url = `${SERVER_BASE_URL}/precision_search`;
      console.log("🎯 [정밀검색] 요청:", url, "| query:", searchText.trim());

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.warn("⚠️ [정밀검색] 서버 응답 실패:", response.status, errorData);
        throw new Error(errorData.detail || '서버 오류');
      }

      const data = await response.json();
      console.log("✅ [정밀검색] 응답 (요약):", {
        stage1: data.stage1_count,
        stage2: data.stage2_evaluated,
        final_text: data.final_text,
        quant_mode: data.quant_mode,
        results_len: Array.isArray(data.results) ? data.results.length : 0,
      });

      setSearchResults(Array.isArray(data.results) ? data.results : []);
      setFinalText(typeof data.final_text === 'string' ? data.final_text : '');
      setStageInfo({
        stage1: typeof data.stage1_count === 'number' ? data.stage1_count : undefined,
        stage2: typeof data.stage2_evaluated === 'number' ? data.stage2_evaluated : undefined,
        quant: typeof data.quant_mode === 'string' ? data.quant_mode : undefined,
      });
      setHasSearched(true);
    } catch (error: any) {
      console.error("❌ [정밀검색] 에러:", error?.message || error);
      Alert.alert('통신 에러', `서버에 연결할 수 없습니다.\n(${error?.message ?? '알 수 없는 오류'})`);
    } finally {
      setIsLoading(false);
    }
  };

  // index.tsx / ai_search.tsx 와 동일한 안전장치
  const openShopLink = async (link: string) => {
    if (!link) {
      Alert.alert('알림', '상품 링크가 없습니다.');
      return;
    }
    let targetUrl = link.trim();
    if (targetUrl.startsWith('//')) {
      targetUrl = 'https:' + targetUrl;
    } else if (!targetUrl.startsWith('http')) {
      targetUrl = 'https://' + targetUrl;
    }
    try {
      await Linking.openURL(targetUrl);
    } catch (e) {
      Alert.alert('오류', '링크를 열 수 없습니다.');
    }
  };

  const getValidImageUrl = (url: string) => {
    if (!url) return 'https://via.placeholder.com/90?text=No+Image';
    let validUrl = url.trim();
    if (validUrl.startsWith('//')) {
      validUrl = 'https:' + validUrl;
    }
    return validUrl;
  };

  const resetSearch = () => {
    setSearchResults([]);
    setHasSearched(false);
    setFinalText('');
    setStageInfo({});
  };

  // multi-head 점수는 raw 합산(보통 0 ~ 12 범위)이므로, 표시용으로 12로 나눠 % 변환
  const formatScore = (mh: any): string => {
    if (typeof mh !== 'number' || !isFinite(mh)) return '-';
    const norm = Math.max(0, Math.min(1, mh / 12));
    return `${(norm * 100).toFixed(1)}%`;
  };

  // ───────── 결과 화면 ─────────
  if (searchResults.length > 0) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.resultContainer}>
          <Text style={styles.searchTitle}>🎯 정밀 검색 결과</Text>

          <View style={styles.metaBox}>
            {finalText ? (
              <>
                <Text style={styles.metaLabel}>AI 해석 (BLIP-2 입력 텍스트)</Text>
                <Text style={styles.metaText}>{finalText}</Text>
              </>
            ) : null}
            <Text style={styles.metaSub}>
              Stage 1: {stageInfo.stage1 ?? '?'}개 후보 · Stage 2: {stageInfo.stage2 ?? '?'}개 평가
              {stageInfo.quant ? `  ·  ${stageInfo.quant}` : ''}
            </Text>
          </View>

          {searchResults.map((item, index) => (
            <View key={index} style={styles.resultCard}>
              <Image
                source={{ uri: getValidImageUrl(item.image_url) }}
                style={styles.resultImage}
                resizeMode="cover"
              />
              <View style={styles.resultInfo}>
                <Text style={styles.resultCategory}>
                  [{item.main_category} {' > '} {item.sub_category}]
                </Text>
                <Text style={styles.resultBrand}>{item.brand_name}</Text>
                <Text style={styles.resultName} numberOfLines={2}>{item.name}</Text>
                <Text style={styles.resultPrice}>
                  {item.price ? `${Number(item.price).toLocaleString()}원` : '가격 정보 없음'}
                </Text>
                <Text style={styles.resultSimilarity}>일치율: {formatScore(item.mh_score)}</Text>
                <TouchableOpacity onPress={() => openShopLink(item.shop_link)}>
                  <Text style={styles.resultLink}>무신사에서 보기 🔗</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))}

          <TouchableOpacity style={styles.resetButton} onPress={resetSearch}>
            <Text style={styles.resetButtonText}>다른 옷 검색하기</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    );
  }

  // ───────── 결과 0개 ─────────
  if (hasSearched && searchResults.length === 0 && !isLoading) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centerContainer}>
          <Text style={styles.placeholderIcon}>🤔</Text>
          <Text style={styles.searchTitle}>일치하는 옷을 찾지 못했어요</Text>
          <Text style={styles.subtitle}>
            요구사항을 더 구체적으로 적어보거나, 다른 사진으로 시도해보세요.
          </Text>
          {finalText ? (
            <View style={[styles.metaBox, { maxWidth: 320, marginTop: 16 }]}>
              <Text style={styles.metaLabel}>AI 해석</Text>
              <Text style={styles.metaText}>{finalText}</Text>
              <Text style={styles.metaSub}>
                Stage 1: {stageInfo.stage1 ?? '?'}개 · Stage 2: {stageInfo.stage2 ?? '?'}개
              </Text>
            </View>
          ) : null}
          <TouchableOpacity style={styles.loginButton} onPress={resetSearch}>
            <Text style={styles.loginButtonText}>다시 검색하기</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ───────── 입력 화면 ─────────
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.mainContent}>
          <Text style={styles.searchTitle}>🎯 AI 정밀검색</Text>
          <Text style={styles.subtitle}>
            빠른 검색 + 정밀 비교로 더 정확하게 찾아드려요{'\n'}(시간이 다소 걸릴 수 있어요)
          </Text>

          <TextInput
            style={styles.textInput}
            placeholder="예) 이 옷과 비슷한데 패턴만 다른 옷"
            value={searchText}
            onChangeText={setSearchText}
            multiline={false}
          />

          <TouchableOpacity style={styles.imageContainer} onPress={pickImage}>
            {imageUri ? (
              <Image source={{ uri: imageUri }} style={styles.image} />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Text style={styles.placeholderIcon}>📷</Text>
                <Text style={styles.placeholderText}>레퍼런스 옷 사진 첨부 (클릭)</Text>
              </View>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={styles.searchButton} onPress={searchClothes} disabled={isLoading}>
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.searchButtonText}>정밀 검색 🎯</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

// ai_search.tsx 의 styles 와 거의 동일 (metaBox 만 추가)
const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#fff' },
  container: { flex: 1, paddingHorizontal: 20 },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 30 },
  mainContent: { flex: 1, justifyContent: 'center' },
  searchTitle: { fontSize: 22, fontWeight: 'bold', marginBottom: 12, color: '#333', textAlign: 'center' },
  subtitle: { fontSize: 14, color: '#666', marginBottom: 24, textAlign: 'center' },
  loginButton: { backgroundColor: '#8B5CF6', paddingVertical: 15, paddingHorizontal: 30, borderRadius: 25, marginTop: 10 },
  loginButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  textInput: { height: 50, borderColor: '#ddd', borderWidth: 1, borderRadius: 10, paddingHorizontal: 15, marginBottom: 20, fontSize: 16, backgroundColor: '#FAFAFA' },
  imageContainer: { height: 250, backgroundColor: '#f9f9f9', borderRadius: 15, borderWidth: 1.5, borderColor: '#ddd', borderStyle: 'dashed', overflow: 'hidden', marginBottom: 20, justifyContent: 'center', alignItems: 'center' },
  imagePlaceholder: { alignItems: 'center' },
  placeholderIcon: { fontSize: 40, marginBottom: 10 },
  placeholderText: { color: '#888', fontSize: 16 },
  image: { width: '100%', height: '100%' },
  searchButton: { backgroundColor: '#8B5CF6', height: 55, borderRadius: 10, justifyContent: 'center', alignItems: 'center' },
  searchButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  resultContainer: { flex: 1, padding: 20 },
  resultCard: { flexDirection: 'row', backgroundColor: '#FAFAFA', borderRadius: 12, padding: 12, marginBottom: 15, borderWidth: 1, borderColor: '#EEE' },
  resultImage: { width: 90, height: 90, borderRadius: 8, marginRight: 15 },
  resultInfo: { flex: 1, justifyContent: 'center' },
  resultCategory: { fontSize: 12, color: '#8B5CF6', fontWeight: 'bold', marginBottom: 4 },
  resultBrand: { fontSize: 13, color: '#333', fontWeight: '600', marginBottom: 2 },
  resultName: { fontSize: 15, fontWeight: '600', color: '#333', marginBottom: 6 },
  resultPrice: { fontSize: 16, fontWeight: 'bold', color: '#333', marginTop: 2, marginBottom: 4 },
  resultSimilarity: { fontSize: 13, color: '#10B981', marginBottom: 6, fontWeight: 'bold' },
  resultLink: { fontSize: 14, color: '#3B82F6', textDecorationLine: 'underline' },
  resetButton: { backgroundColor: '#333', height: 50, borderRadius: 10, justifyContent: 'center', alignItems: 'center', marginTop: 10, marginBottom: 40 },
  resetButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },

  // 메타 박스 — Gemini 해석 + stage 통계 함께 노출
  metaBox: { backgroundColor: '#F5F3FF', borderRadius: 8, padding: 12, marginBottom: 16, borderWidth: 1, borderColor: '#E9D5FF' },
  metaLabel: { fontSize: 11, color: '#8B5CF6', fontWeight: 'bold', marginBottom: 4 },
  metaText: { fontSize: 13, color: '#444', lineHeight: 18, marginBottom: 6 },
  metaSub: { fontSize: 11, color: '#888' },
});
