from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(data):
    # Pisahkan fitur (X) dan target (y)
    X = data.drop('price', axis=1)
    y = data['price']

    # Bagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model regresi linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Simpan daftar fitur ke file
    joblib.dump(list(X.columns), 'features.pkl')

    # Evaluasi model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return model
