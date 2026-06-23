import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    "hotel_bookings_raw.csv"
)

df.drop(
    columns=["company","agent"],
    inplace=True
)

df.dropna(
    inplace=True
)

df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include='object'):

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

df.to_csv(
    "preprocessing/hotel_bookings_preprocessing.csv",
    index=False
)

print("Preprocessing selesai")