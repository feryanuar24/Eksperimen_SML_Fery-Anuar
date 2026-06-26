import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Data
df = pd.read_csv(
    "hotel-booking_raw.csv"
)

# Hapus Fitur
df.drop(
    columns=["company","agent", "reservation_status"],
    inplace=True
)

# Hapus Missing Value
df.dropna(
    inplace=True
)

# Hapus Duplikat
df.drop_duplicates(inplace=True)

# Encoding Fitur Kategorikal
for col in df.select_dtypes(include='object'):

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

# Simpan Data Hasil Preprocessing
df.to_csv(
    "preprocessing/hotel-booking_preprocessing.csv",
    index=False
)

print("Preprocessing selesai")