import pandas as pd
import io
import os
import pytz

pm10_file_paths = [
    "C:\\Users\\LENOVO\\OneDrive\\Pictures\\air-pm10-forecast\\src"
]
poland_timezone = pytz.timezone("Europe/Warsaw")
valid_range = (0, 1000)

""" Load and merge data from multiple PM10 Excel files."""

def load_pm10_files(file_paths):
    dfs = []
    for file in file_paths:
        df = pd.read_excel(file)
        df['source_file'] = os.path.basename(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

combined_df = load_pm10_files(pm10_file_paths)

""" Clean data by removing missing and duplicate. """

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df.sort_values(by='DateTime').reset_index(drop=True)

combined_df = clean_data(combined_df)

""" Detect anomalies values outside the valid range."""

def detect_anomalies(df, data_columns, min_val, max_val):
    outsideRange = (df[data_columns] < min_val) | (df[data_columns] > max_val)
    anomalies = outsideRange.sum()
    print("Anomalies:\n", anomalies)
    return df[outsideRange.any(axis=1)]

data_columns = combined_df.columns[1:8]
anomalies_rows = detect_anomalies(combined_df, data_columns, *valid_range)

""" Convert local time to UTC """

def convert_datetime_to_utc(df):
    df['DateTime'] = df['DateTime'].dt.tz_localize(
        poland_timezone, ambiguous='NaT', nonexistent='shift_forward'
    ).dt.tz_convert('UTC')
    return df

combined_df = convert_datetime_to_utc(combined_df)

""" Ensure that all text columns are UTF-8 encoded. """

def encoding_format(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).apply(
            lambda x: x.encode('utf-8', 'replace').decode('utf-8', 'replace')
        )
    return df
combined_df = encoding_format(combined_df)

combined_df.head()
