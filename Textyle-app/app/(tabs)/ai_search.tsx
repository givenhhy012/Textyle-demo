import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import { ActivityIndicator, Alert, Image, Linking, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

/**
 * AI 검색 화면.
 * 사용자: 옷 이미지 + 텍스트(요구사항) → 서버가 Gemini 로 의도를 융합하여 검색.
 * 백엔드: Textyle-vectorserver/gemini_search_main.py (port 8001) 의 POST /ai_search
 */
export default function AiSearchScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [searchText, setSearchText] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [hasSearched, setHasSearched] = useState(false);

  // 서버가 응답에 함께 내려주는 디버그 정보 — 사용자가 LLM 해석을 확인할 수 있도록 노출
  const [finalText, setFinalText] = useState<string>('');
  const [imageAttrs, setImageAttrs] = useState<Record<string, string> | null>(null);

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
    setImageAttrs(null);

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

      // 🚨 본인 PC IP 확인! 기존 index.tsx 의 SERVER_IP 와 동일 값 사용.
      const SERVER_IP = "192.168.0.40";
      const url = `http://${SERVER_IP}:8001/ai_search`;
      console.log("🔎 [AI 검색] 요청:", url, "| query:", searchText.trim());

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.warn("⚠️ [AI 검색] 서버 응답 실패:", response.status, errorData);
        throw new Error(errorData.detail || '서버 오류');
      }

      const data = await response.json();
      console.log("✅ [AI 검색] 응답:", data);

      setSearchResults(Array.isArray(data.results) ? data.results : []);
      setFinalText(typeof data.final_text === 'string' ? data.final_text : '');
      setImageAttrs(
        data.image_attributes && typeof data.image_attributes === 'object'
          ? data.image_attributes
          : null,
      );
      setHasSearched(true);
    } catch (error: any) {
      console.error("❌ [AI 검색] 에러:", error?.message || error);
      Alert.alert('통신 에러', `서버에 연결할 수 없습니다.\n(${error?.message ?? '알 수 없는 오류'})`);
    } finally {
      setIsLoading(false);
    }
  };

  // index.tsx 와 동일한 안전장치 함수들 (코드 재사용)
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
    setImageAttrs(null);
  };

  // ───────── 결과 화면 ─────────
  if (searchResults.length > 0) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.resultContainer}>
          <Text style={styles.searchTitle}>✨ AI가 찾아낸 옷이에요!</Text>

          {finalText ? (
            <View style={styles.debugBox}>
              <Text style={styles.debugLabel}>AI 해석</Text>
              <Text style={styles.debugText}>{finalText}</Text>
            </View>
          ) : null}

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
                <Text style={styles.resultSimilarity}>
                  일치율: {typeof item.similarity === 'number' ? (item.similarity * 100).toFixed(1) : '-'}%
                </Text>
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

  // ───────── 결과 0개 화면 ─────────
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
            <View style={[styles.debugBox, { maxWidth: 320 }]}>
              <Text style={styles.debugLabel}>AI 해석</Text>
              <Text style={styles.debugText}>{finalText}</Text>
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
          <Text style={styles.searchTitle}>✨ AI 패션 검색</Text>
          <Text style={styles.subtitle}>
            사진과 요청사항을 함께 보내면 AI가 의도를 이해해서 옷을 찾아드려요
          </Text>

          <TextInput
            style={styles.textInput}
            placeholder="예) 이 옷과 비슷한데 색이 파란색인 옷"
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
              <Text style={styles.searchButtonText}>AI로 찾아보기 ✨</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

// 스타일 — index.tsx 와 거의 동일, debugBox 만 추가
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

  // 디버그 박스 — AI 해석 결과를 노출 (계획서의 "선택" 항목)
  debugBox: { backgroundColor: '#F5F3FF', borderRadius: 8, padding: 12, marginBottom: 16, borderWidth: 1, borderColor: '#E9D5FF' },
  debugLabel: { fontSize: 11, color: '#8B5CF6', fontWeight: 'bold', marginBottom: 4 },
  debugText: { fontSize: 13, color: '#444', lineHeight: 18 },
});
