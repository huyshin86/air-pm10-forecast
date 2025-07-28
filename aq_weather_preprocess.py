#!/usr/bin/env python3
"""
Convert Krakow AQ & weather data to the JSON format required by the PM10 model.
Creates daily “cases” (2019‑2023) whose prediction_start_time is D + 2 days.
Columns with ≥ 50 % NaNs are dropped. Weather groups are richly decoded.
"""

import json, logging, re
from math import exp
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ──────────────────────────── regex & helper functions ─────────────────────────

_temp_re  = re.compile(r'^[\+\-]?(\d{4,5})')   # "+0001,1" → "0001"
_num_re   = re.compile(r'^(\d+)')              # leading numeric string
# --- add helper for generic 3‑digit -> int but skip "999" sentinel
def _three_digit(raw):
    m = _num_re.match(str(raw))
    if m and m.group(1) != "999":
        return int(m.group(1))
    return None

def _split_all(raw: str) -> list[str]:
    """Return list even if raw is nan/None."""
    return str(raw).split(',') if pd.notna(raw) else []

def _parse_temp(raw: str) -> float | None:
    m = _temp_re.match(str(raw))
    return int(m.group(1)) / 10.0 if m else None

def _parse_vis(raw: str) -> int | None:
    m = _num_re.match(str(raw))
    return int(m.group(1)) if m else None

def _parse_wind(raw: str) -> tuple[float | None, float | None]:
    try:
        d, *_ , s, _ = str(raw).split(',')
        return float(d), int(s) / 10.0
    except Exception:
        return None, None

def _rel_humidity(t: float, td: float) -> float | None:
    try:
        es = 6.1094 * exp(17.625 * t  / (t  + 243.04))
        e  = 6.1094 * exp(17.625 * td / (td + 243.04))
        return round((e / es) * 100, 1)
    except Exception:
        return None

def _parse_cig(raw):            # ceiling height → metres
    m = _num_re.match(str(raw))
    return int(m.group(1)) * 0.3048 if m and m.group(1) != "99999" else None

def _parse_precip(raw):         # hundredths‑inch → mm
    m = _num_re.match(str(raw))
    return round(int(m.group(1)) * 0.254, 1) if m else None

def _parse_pressure_tendency(raw):
    try:
        code, *_ , delta, _ = str(raw).split(',')
        return int(code[0]), round(int(delta) / 10.0, 1)
    except Exception:
        return None, None

# ────────────────────────────────── main class ─────────────────────────────────

class DataConverter:
    """Turn raw Excel/CSV into model‑ready JSON cases."""

    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir        = Path(data_dir)
        self.air_quality_dir = self.data_dir / "AirQuality_Krakow"
        self.weather_dir     = self.data_dir / "Weather_Krakow"

    # ───── utilities ─────
    @staticmethod
    def _prune_columns(df: pd.DataFrame, time_col='DateTime'):
        null_frac = df.isna().mean()
        keep      = [c for c in df.columns
                     if c == time_col or null_frac[c] < 0.50]
        dropped = [c for c in df.columns if c not in keep]
        if dropped:
            log.debug(f"Dropping {len(dropped)} sparse cols: {dropped[:6]}…")
        return df[keep]

    # ───── file loaders ─────
    def load_stations(self):
        f = self.air_quality_dir / "Stations.xlsx"
        if f.exists():
            return pd.read_excel(f)
        log.warning("Stations.xlsx not found – default coords will be used.")
        return None

    def load_air_quality_data(self, year: int):
        fp = self.air_quality_dir / f"{year}_PM10_1g.xlsx"
        if not fp.exists():
            log.error(f"AQ file not found: {fp}")
            return None
        df = pd.read_excel(fp)
        ts_col = 'DateTime' if 'DateTime' in df.columns else 'DATE'
        df[ts_col] = pd.to_datetime(df[ts_col])
        if ts_col == 'DATE':
            df = df.rename(columns={'DATE': 'DateTime'})
        return self._prune_columns(df)

    def load_weather_data(self, year: int):
        fp = self.weather_dir / f"{year}.csv"
        if not fp.exists():
            log.error(f"Weather file not found: {fp}")
            return None
        df = pd.read_csv(fp, low_memory=False)
        df['DateTime'] = pd.to_datetime(df['DATE'])
        df = df.drop(columns=['DATE'])
        return self._prune_columns(df)
    
 


    # ───── weather row encoder ─────
    def encode_weather(self, row) -> dict:
        rec: dict = {}

        # ── timestamp ─────────────────────────────
        rec['date'] = row['DateTime'].strftime('%Y-%m-%dT%H:%M:%S')

        # ── TMP (temperature) ─────────────────────
        if 'TMP' in row and pd.notna(row['TMP']):
            t_parts = _split_all(row['TMP'])
            t_val   = _parse_temp(t_parts[0])
            if t_val is not None:
                rec['tmp_c']  = t_val
                if len(t_parts) > 1 and t_parts[1] != ' ':        # quality flag
                    rec['tmp_qc'] = int(t_parts[1])

        # ── DEW (dew‑point) ───────────────────────
        if 'DEW' in row and pd.notna(row['DEW']):
            d_parts = _split_all(row['DEW'])
            d_val   = _parse_temp(d_parts[0])
            if d_val is not None:
                rec['dew_c'] = d_val

        # RH
        if 'tmp_c' in rec and 'dew_c' in rec:
            rh = _rel_humidity(rec['tmp_c'], rec['dew_c'])
            if rh is not None:
                rec['rel_hum'] = rh

        # ── WND (wind) ────────────────────────────
        if 'WND' in row and pd.notna(row['WND']):
            w = _split_all(row['WND'])                 # dir, qc1, var_code, spd, qc2
            if len(w) >= 5:
                dir_deg        = float(w[0]) if w[0].isdigit() else None
                spd_ms         = int(w[3]) / 10.0      # 1 tenth m s⁻¹
                rec['wnd_dir_deg']  = dir_deg
                rec['wnd_speed_ms'] = spd_ms
                rec['wind_qc_dir']  = int(w[1])
                rec['wind_var_code']= w[2]
                rec['wind_qc_spd']  = int(w[4])

        # ── SLP (sea‑level pressure) ───────────────
        if 'SLP' in row and pd.notna(row['SLP']):
            s_parts =_split_all(row['SLP'])           # value, qc
            if s_parts[0] != '99999':
                rec['pressure_hpa'] = int(s_parts[0]) / 10.0
                if len(s_parts) > 1:
                    rec['slp_qc'] = int(s_parts[1])

        # ── VIS (visibility) ───────────────────────
        if 'VIS' in row and pd.notna(row['VIS']):
            v = _split_all(row['VIS'])                 # dist, qc, variability, qc2
            vis_val = _parse_vis(v[0])
            if vis_val is not None:
                rec['vis_m'] = vis_val
            if len(v) > 1:
                rec['vis_qc'] = int(v[1])
            if len(v) > 2 and v[2].isdigit():
                rec['vis_variability'] = int(v[2])
        # ── GA1‑GA3  (past weather codes) ─────────────────────────────
        for tag, key in (('GA1', 'wx_past_3h_1'),
                        ('GA2', 'wx_past_3h_2'),
                        ('GA3', 'wx_past_3h_3')):
            if tag in row and pd.notna(row[tag]):
                code = _three_digit(row[tag])
                if code is not None:
                    rec[key] = code

        # ── GE1  (sunshine minutes) ──────────────────────────────────
        if 'GE1' in row and pd.notna(row['GE1']):
            sun = _three_digit(row['GE1'])
            if sun is not None:
                rec['sunshine_min'] = sun        # already had this earlier; keeps it

        # ── GF1  (total cloud cover) ─────────────────────────────────
        if 'GF1' in row and pd.notna(row['GF1']):
            cc = _three_digit(row['GF1'])
            if cc is not None:
                rec['cloud_tot_oktas'] = cc      # 0‑8 oktas; 9=sky obscured

        # ── MA1  (max surface temp last 6 h) ─────────────────────────
        if 'MA1' in row and pd.notna(row['MA1']):
            t = _parse_temp(row['MA1'])
            if t is not None:
                rec['max_surf_temp_c'] = t

        # ── REM  (free‑text remarks) ─────────────────────────────────
        if 'REM' in row and pd.notna(row['REM']):
            rec['remarks'] = str(row['REM']).strip()

        # ── everything else stays (cig, precip, codes, etc.) ────
        # … keep the rest of your existing parsing blocks unchanged …

        # ── calendar context ──────────────────────
        ts = row['DateTime']
        rec['month'] = ts.month
        rec['doy']   = ts.dayofyear
        rec['hour']  = ts.hour
        return rec


    # ───── main case‑builder ─────
    def create_training_cases(self, years=(2019, 2020, 2021, 2022, 2023)):
        cases, case_id = [], 1
        stations_df = self.load_stations()

        for yr in years:
            log.info(f"Processing {yr} …")
            aq, wx = self.load_air_quality_data(yr), self.load_weather_data(yr)
            if aq is None or wx is None:
                log.warning(f"Skipping {yr} – missing data.")
                continue

            for day in pd.date_range(f'{yr}-01-01', f'{yr}-12-29'):  # leave 2d gap
                one_day = pd.Timedelta(days=1)

                aq_day = aq[(aq['DateTime'] >= day) & (aq['DateTime'] < day + one_day)]
                wx_day = wx[(wx['DateTime'] >= day) & (wx['DateTime'] < day + one_day)]

                if len(aq_day) < 12:        # ≥ 12h AQ data
                    continue

                case = {
                    "case_id": f"case_{case_id:04d}",
                    "stations": [],
                    "target": {
                        "longitude": 19.926189,
                        "latitude" : 50.057678,
                        "prediction_start_time": (day + timedelta(days=2)).strftime('%Y-%m-%dT00:00:00')
                    },
                    "weather": []
                }

                for col in [c for c in aq_day.columns if c != 'DateTime']:
                    if aq_day[col].isna().mean() >= 0.50:
                        continue                      # drop extra‑sparse stations
                    hist = [{"timestamp": t.strftime('%Y-%m-%dT%H:%M:%S'),
                             "pm10": v}
                            for t, v in zip(aq_day['DateTime'], aq_day[col]) if pd.notna(v)]
                    if len(hist) < 12:
                        continue
                    lat, lon = 50.057678, 19.926189
                    if stations_df is not None:
                        match = stations_df[stations_df['Station Code'] == col.replace('PM10_', '')]
                        if match.empty and col.startswith('Mp'):
                            match = stations_df[stations_df['Station Code'] == col]
                        if not match.empty:
                            lat = float(match.iloc[0]['WGS84 φ N'])
                            lon = float(match.iloc[0]['WGS84 λ E'])
                    case['stations'].append({"station_code": col,
                                             "latitude"    : lat,
                                             "longitude"   : lon,
                                             "history"     : hist})

                case['weather'] = [self.encode_weather(r) for _, r in wx_day.iterrows()
                                   if r.notna().any()]
                if case['stations'] and case['weather']:
                    cases.append(case); case_id += 1
                    if case_id % 100 == 0:
                        log.info(f"  … {case_id} cases so far")

        log.info(f"Finished: {case_id-1} cases.")
        return cases

    # ───── orchestration ─────
    def convert_and_save(self, out="krakow_training_data.json",
                         years=(2019, 2020, 2021, 2022, 2023)):
        log.info("Starting conversion …")
        cases = self.create_training_cases(years)
        Path(out).write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False))
        log.info(f"Saved {len(cases)} cases → {out}")
        return cases

# ──────────────────────────────── runner ──────────────────────────────────────
if __name__ == "__main__":
    conv  = DataConverter()
    cases = conv.convert_and_save()
    print(f"✅  {len(cases)} cases written to krakow_training_data.json")
    if cases:
        print(f"   Sample case_id: {cases[0]['case_id']}  "
              f"stations: {len(cases[0]['stations'])}  "
              f"weather rows: {len(cases[0]['weather'])}")
