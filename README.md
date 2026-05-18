# Enerji Tüketimi Tahminlemesi (Energy Consumption Forecasting)

Bu proje, PJME (PJM East Region) saatlik enerji tüketim veri seti kullanılarak, derin öğrenme tabanlı LSTM (Long Short-Term Memory) modelleri ile Haftalık, Aylık ve Yıllık enerji tüketim tahminleri yapmak amacıyla ENM421 Veri Bilimine Giriş dersi kapsamında geliştirilmiştir.

## Proje Amacı ve Problem Tanımı
Elektrik üretim ve tüketim dengesini sağlamak şebeke yönetimi için hayati önem taşır. Yanlış tahminler elektrik kesintilerine, israfa veya dışarıdan yüksek maliyetli enerji alımına neden olabilir. Bu projenin amacı, karmaşık mevsimsellik barındıran zaman serisi (Time Series) enerji verisinden yararlanarak gelecekteki makro tüketim trendlerini güvenilir bir şekilde öngörmektir.

## Veri Seti ve Keşifsel Veri Analizi (EDA)
- **Veri Seti:** PJME_hourly.csv (2002-2018 yılları arası saatlik tüketim, Megawatt)
- Saatlik bazdaki verilerin oluşturduğu anlık "gürültüyü" (noise) azaltmak ve LSTM modelinin makro trendleri görebilmesini sağlamak amacıyla veri **"Günlük Ortalamalara (Daily Mean)"** indirgenmiştir.

## Veri Temizleme ve Ön İşleme
- **Eksik Veri Yönetimi:** Eksik günlerin temizlenmesi (dropna) ve tarih indeksinin kronolojik olarak sıralanması.
- **Ölçeklendirme (Scaling):** Derin öğrenme algoritmalarının daha hızlı ve stabil çalışabilmesi (Yakınsama / Convergence) için veriler `MinMaxScaler` ile (0, 1) aralığına sıkıştırıldı.
- **Pencereleme (Windowing):** Modelin girdi olarak alacağı geçmiş veri (Look-back) ile tahmin edeceği ufuk (Forecast Horizon) ayrıştırılmıştır. Eğitim için veri %80, test için %20 oranında bölünmüştür.

## Kullanılan Yöntemler ve Makine Öğrenmesi Algoritması
Projede **Derin Öğrenme** algoritmalarından olan **LSTM (Long Short-Term Memory)** kullanılmıştır.
Geleneksel istatistiksel modellere kıyasla, LSTM ağları içlerindeki "Kapı (Gate)" mekanizmaları sayesinde uzun süreli bağımlılıkları (Long-term dependencies) en iyi şekilde modelleyebilir. 
Model Mimarisi:
- İki adet LSTM katmanı (128 ve 64 Nöron, 'tanh' aktivasyon)
- Aşırı öğrenmeyi engellemek için %20 oranında Dropout
- Gradient Patlamasını önlemek adına Gradient Clipping (clipvalue=1.0)
- EarlyStopping ve ReduceLROnPlateau teknikleri ile dinamik optimizasyon

## Özet Sonuçlar ve Performans Metrikleri
Model test seti üzerinde çalıştırılmış ve aşağıdaki hata payları elde edilmiştir:
- **Haftalık (7 Gün):** MAPE ~%6.69
- **Aylık (30 Gün):** MAPE ~%9.14
- **Yıllık (365 Gün):** MAPE ~%9.35

Model, kısa vadede oldukça isabetli sonuçlar verirken, uzun vadeli (yıllık) tahminlerde genel yaz/kış mevsimsel dalgalanmalarını (makro trendleri) başarıyla simüle etmiştir.
