import pandas as pd

def preprocess_data(file_path):
    # Baca dataset
    data = pd.read_csv(file_path)

    # Konversi kolom 'date' ke tipe datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        # Ekstraksi fitur dari kolom 'date'
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['day_of_week'] = data['date'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        # Hapus kolom 'date'
        data = data.drop(columns=['date'])

    # Hapus kolom non-numerik yang tidak relevan
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    data = data.drop(columns=non_numeric_cols)

    # Pastikan tidak ada nilai NaN atau infinite
    data = data.dropna()

    return data
