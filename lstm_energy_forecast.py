import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

def load_and_preprocess_data(file_path):
    print("Veri yükleniyor...")
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.sort_index()
    
    # Günlük ortalamaya çevirerek veriyi sadeleştiriyoruz
    # Böylece haftalık, aylık, yıllık tahminleri daha rahat yapabiliriz
    df_daily = df.resample('D').mean()
    df_daily = df_daily.dropna()
    return df_daily

def create_sequences(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + forecast_horizon)])
    return np.array(X), np.array(y)

def build_lstm_model(look_back, forecast_horizon):
    model = Sequential()
    # Kapasiteyi artırdık: 64 -> 128 ve 32 -> 64
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_horizon))
    
    # clipvalue=1.0 ekleyerek gradient patlamasının (NaN sorunu) önüne geçiyoruz
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_and_predict(df_daily, look_back, forecast_horizon, title, epochs=100, batch_size=32):
    print(f"--- {title} Tahmini İçin Model Eğitiliyor ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_daily)
    
    X, y = create_sequences(scaled_data, look_back, forecast_horizon)
    
    # Train-test split (Son %20 test)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_lstm_model(look_back, forecast_horizon)
    
    # Modele Early Stopping ekleyerek loss düşmeyi bıraktığında eğitimi durduralım
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    
    # Öğrenme oranını (Learning Rate) zamanla küçülterek daha hassas noktaya ulaşmasını sağlayalım
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop, reduce_lr], verbose=1)
    
    # --- TEST VS TAHMİN KARŞILAŞTIRMASI ---
    # Test setindeki en son veriyi kullanarak modelimizin test başarısını çizelim
    idx = -1 # Test setinin en son örneği
    test_pred_scaled = model.predict(X_test[idx].reshape(1, look_back, 1))
    
    actual_test_sample = scaler.inverse_transform(y_test[idx].reshape(1, -1))[0]
    pred_test_sample = scaler.inverse_transform(test_pred_scaled)[0]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, forecast_horizon + 1), actual_test_sample, label='Gerçek Test Verisi (Actual)', marker='o')
    plt.plot(range(1, forecast_horizon + 1), pred_test_sample, label='Model Tahmini (Predicted)', marker='x', linestyle='dashed')
    plt.title(f'{title} Tahmini: Test Verisi vs Model Tahmini')
    plt.xlabel('Gelecek Günler (Test Periyodu)')
    plt.ylabel('Megawatt (MW)')
    plt.legend()
    plt.grid(True)
    os.makedirs('charts', exist_ok=True)
    plt.savefig(f'charts/{title.lower().replace(" ", "_")}_test_vs_pred.png')
    plt.show()
    
    # --- PERFORMANS METRİKLERİ ---
    mae = mean_absolute_error(actual_test_sample, pred_test_sample)
    rmse = np.sqrt(mean_squared_error(actual_test_sample, pred_test_sample))
    
    # MAPE (Ortalama Mutlak Yüzde Hata)
    # Gerçek veri 0 olamayacağı için (enerji tüketimi) direkt bölebiliriz
    mape = np.mean(np.abs((actual_test_sample - pred_test_sample) / actual_test_sample)) * 100
    
    print(f"\n--- {title} Performans Metrikleri ---")
    print(f"MAE (Ortalama Mutlak Hata): {mae:.2f} MW")
    print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.2f} MW")
    print(f"MAPE (Ortalama Mutlak Yüzde Hata): %{mape:.2f}\n")
    
    with open("performans_metrikleri.txt", "a") as f:
        f.write(f"[{title} Tahmini]\nMAE: {mae:.2f} MW\nRMSE: {rmse:.2f} MW\nMAPE: %{mape:.2f}\n\n")
        
    # --- GELECEK BİLİNMEYEN UZAY TAHMİNİ ---
    # Son 'look_back' günü alıp gelecek (tamamen bilinmeyen) 'forecast_horizon' gününü tahmin edelim
    last_sequence = scaled_data[-look_back:]
    last_sequence = last_sequence.reshape((1, look_back, 1))
    
    future_pred_scaled = model.predict(last_sequence)
    future_pred = scaler.inverse_transform(future_pred_scaled)[0]
    
    plt.figure(figsize=(10, 5))
    last_actual_days = df_daily.iloc[-look_back:].index
    future_dates = pd.date_range(start=last_actual_days[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    
    plt.plot(last_actual_days, df_daily.iloc[-look_back:].values, label='Gerçek Veri (Son Günler)')
    plt.plot(future_dates, future_pred, label=f'Gelecek {forecast_horizon} Gün Tahmini', marker='o')
    plt.title(f'Enerji Tüketimi {title} Gelecek Tahmini (LSTM)')
    plt.xlabel('Tarih')
    plt.ylabel('Megawatt (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'charts/{title.lower().replace(" ", "_")}_future_forecast.png')
    plt.show()
    
    print(f"{title} tahmini tamamlandı ve grafikler charts/ klasörüne kaydedildi.\n")

if __name__ == '__main__':
    # Eski metrikleri temizle
    if os.path.exists("performans_metrikleri.txt"):
        os.remove("performans_metrikleri.txt")
        
    # Veri seti yolu
    file_path = 'archive/PJME_hourly.csv'
    
    if not os.path.exists(file_path):
        print(f"Hata: {file_path} bulunamadı!")
    else:
        df_daily = load_and_preprocess_data(file_path)
        
        # 1. Haftalık Tahmin (7 gün sonrası)
        train_and_predict(df_daily, look_back=60, forecast_horizon=7, title="Haftalik", epochs=100)
        
        # 2. Aylık Tahmin (30 gün sonrası)
        train_and_predict(df_daily, look_back=90, forecast_horizon=30, title="Aylik", epochs=100)
        
        # 3. Yıllık Tahmin (365 gün sonrası)
        train_and_predict(df_daily, look_back=365, forecast_horizon=365, title="Yillik", epochs=100)
