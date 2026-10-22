import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

hubs = pd.read_csv("Capstone_Project/data/hubs.csv")
print(hubs.head())

raw_ship = pd.read_csv("Capstone_Project/data/raw_shipments.csv")
print(raw_ship.head())
print(raw_ship.duplicated().sum())

new_ship = pd.read_csv("Capstone_Project/data/new_shipments.csv")
print(new_ship.head())

print(PROMISED_HOURS)
print(FEATURES)

def clean_shipement(raw:pd.DataFrame,medians:dict=None)->tuple[pd.DataFrame,dict]:
    """Cleans raw shipment data and handles missing values using medians imputation"""
    df = raw.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    df["service_type"]=df["service_type"].astype(str).str.strip().str.title()

    df['payment_mode'] = df["payment_mode"].fillna("Unkonwn")

    df["declared_value_inr"]=(
        df["declared_value"]
        .astype(str)
        .str.replace("₹","",regex=False)
        .str.replace(",","",regex=False)
        .astype(float)
    )

    df = df.drop(columns=["declared_value"])

    df["dispatch_time"]=pd.to_datetime(df["dispatch_time"])

    df.loc[df["parcel_weight_kg"]>100,"parcel_weight_kg"]=np.nan

    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weighr_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }

    df["distance_km"]=df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"]=df["parcel_weight_kg"].fillna(medians["parcel_weighr_kg"])
    df["declared_value_inr"]=df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians

def build_features(clean:pd.DataFrame,hubs:pd.DataFrame)->pd.DataFrame:
    """Attaches hub metadata and engineers required predictive feature columns"""

    df=clean.merge(hubs,on="hub_id",how="left")

    df["promised_hours"]=df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"]=(df["payment_mode"]=="COD").astype(int)
    df["is_weekend"]=(df["dispatch_time"].dt.dayofweek>=5).astype(int)
    df["dispatch_hour"]=df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"]=np.log(df["declared_value_inr"])

    return df