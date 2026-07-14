#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-4 (v3.0) — Final Feature-Complete + Clean
-----------------------------------------------
• Unicode/tab/boşluk temizliği
• polymer & filler normalization
• family, additive, modification strategy
• numeric winsorization, log features, binning
• missing flags + final NaN fill
• CatBoost’a doğrudan hazır CSV
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ===============================
# CONFIG
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUT_DIR = OUTPUTS_DIR / "stage4_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INP = OUTPUTS_DIR / "results_stage3_verified_dataset.csv"
OUT = OUT_DIR / "results_stage4_features.csv"

# ===============================
# LOAD & CLEAN
# ===============================
with open(INP, "r", encoding="utf-8") as f:
    first_line = f.readline()
sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
print("[i] Detected delimiter:", "TAB" if sep == "\t" else "COMMA")

df = pd.read_csv(INP, sep=sep, engine="python")

# --- sütun ve hücre temizliği ---
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace(u'\xa0', np.nan, regex=True)
df = df.replace(u'\u200b', np.nan, regex=True)
df = df.replace("NaN", np.nan)

# --- text bazlı boşluklar ---
def to_nan(x):
    s = str(x).strip().lower()
    return np.nan if s in ["", "unknown", "not reported", "-", "none", "nan"] else x
for c in df.columns:
    df[c] = df[c].apply(to_nan)

# --- sayısal sütunlar ---
num_cols = ["temperature_C","pressure_bar","filler_loading_wt%","permeability_Barrer","selectivity"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ===============================
# COMPATIBILITY
# ===============================
if "compatibility" in df.columns:
    df["compatibility"] = df["compatibility"].replace("Moderate", "Non-Compatible")

# ===============================
# NORMALIZATION
# ===============================
def normalize_name(name):
    if pd.isna(name): return np.nan
    s = str(name).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("®","").replace("™","").replace("–","-").replace("—","-").replace("‐","-")
    s = re.sub(r"[^a-z0-9\-\s_/()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    match = re.search(r"\(([^)]+)\)", s)
    if match:
        inner = match.group(1)
        if re.search(r"[a-z]{2,3}-?\d+", inner): s = inner
        else: s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("_","-").replace("/","-").replace("\\","-")
    s = re.sub(r"-+","-", s)
    if any(k in s for k in ["blend","mix","copolymer"]):
        s = s + " (blend)"
    return s.strip()

if "polymer" in df.columns: df["polymer_clean"] = df["polymer"].apply(normalize_name)
if "filler"  in df.columns: df["filler_clean"]  = df["filler"].apply(normalize_name)

# ===============================
# FAMILY MAPS
# ===============================
def polymer_family_map(p):
    if pd.isna(p): return "Other"
    if any(x in p for x in ["matrimid","6fda","pim","pei","kapton","ultem"]): return "Polyimide"
    if any(x in p for x in ["pebax","polyether"]): return "Polyether"
    if any(x in p for x in ["psf","pes","psu","ulfone","sulfone"]): return "Polysulfone"
    if any(x in p for x in ["peek"]): return "PEEK"
    if any(x in p for x in ["pvdf","ptfe","teflon"]): return "Fluoropolymer"
    if any(x in p for x in ["pva","pvp","pvoh"]): return "PVA"
    if any(x in p for x in ["pan"]): return "PAN"
    if any(x in p for x in ["pmma","acrylic"]): return "Acrylic"
    if any(x in p for x in ["pc"]): return "Polycarbonate"
    if any(x in p for x in ["ps"]): return "Polystyrene"
    if any(x in p for x in ["ppo","polyphenylene"]): return "PPO"
    if any(x in p for x in ["ptmsp","pdms","silicone"]): return "Silicone"
    return "Other"

def filler_family_map(f):
    if pd.isna(f): return "Other"
    if any(k in f for k in ["zif","uio","mof","mil","hkust"]): return "MOF"
    if any(k in f for k in ["ldh","hydroxide"]): return "LDH"
    if any(k in f for k in ["cnt","mwcnt","graphene","carbon","go"]): return "CarbonBased"
    if any(k in f for k in ["sio2","tio2","zno","al2o3","silica","oxide"]): return "Oxide"
    if any(k in f for k in ["zsm","mcm","sba","zeolite"]): return "Zeolite"
    if any(k in f for k in ["cof"]): return "COF"
    return "Other"

df["polymer_family"] = df["polymer_clean"].apply(polymer_family_map)
df["filler_family"]  = df["filler_clean"].apply(filler_family_map)

# ===============================
# ADDITIVE TYPE & MODIFICATION
# ===============================
def additive_type_from_text(row):
    f = (
        str(row.get("filler_clean") or "") + " " +
        str(row.get("polymer_clean") or "") + " " +
        str(row.get("notes") or "")
    ).lower()
    if any(k in f for k in ["2-ph","aptes","eda","teos","amine","functionalized"]): return "SurfaceModifier"
    if any(k in f for k in ["peg","pvp","pva","polyethylene glycol"]): return "PolymerAdditive"
    if any(k in f for k in ["ionic liquid","il","bmim","hmim"]): return "IonicLiquid"
    if any(k in f for k in ["zif","mof","uio","mil","hkust"]): return "MOF"
    if any(k in f for k in ["cnt","graphene","carbon","go"]): return "CarbonBased"
    if any(k in f for k in ["sio2","tio2","zno","al2o3","oxide"]): return "Oxide"
    if any(k in f for k in ["zsm","mcm","sba","zeolite"]): return "PorousSilicate"
    if any(k in f for k in ["deep eutectic","des"]): return "DES"
    return "Other"

df["additive_type"] = df.apply(additive_type_from_text, axis=1)

def has_text(x):
    return isinstance(x, str) and x.strip().lower() not in ["", "nan"]

def mod_strategy(fm_val, pm_val):
    f = has_text(fm_val); p = has_text(pm_val)
    if not f and not p: return "Pristine"
    if f and not p: return "FillerOnly"
    if not f and p: return "PolymerOnly"
    if f and p: return "DualModified"
    return "Hybrid"

df["modification_strategy"] = [
    mod_strategy(df.get("filler_modification")[i] if "filler_modification" in df else None,
                 df.get("polymer_modification")[i] if "polymer_modification" in df else None)
    for i in range(len(df))
]

# ===============================
# NUMERIC LOG & BINS
# ===============================
for c in ["pressure_bar","permeability_Barrer","selectivity"]:
    if c in df.columns:
        df.loc[df[c] <= 0, c] = np.nan
        low, high = df[c].quantile([0.005, 0.995])
        df[c] = df[c].clip(lower=low, upper=high)

df["normalized_loading"] = df["filler_loading_wt%"] / 100.0

def safe_log10(x): return np.log10(x) if pd.notna(x) and x > 0 else np.nan
if "pressure_bar" in df: df["pressure_log"] = df["pressure_bar"].apply(safe_log10)
if "permeability_Barrer" in df: df["log_perm"] = df["permeability_Barrer"].apply(safe_log10)
if "selectivity" in df: df["log_sel"] = df["selectivity"].apply(safe_log10)
if {"permeability_Barrer","selectivity"}.issubset(df.columns):
    df["PI"] = df["permeability_Barrer"] * df["selectivity"]
    df["PI_log"] = df["PI"].apply(safe_log10)

df["Robeson_ratio"] = df["selectivity"] / df["permeability_Barrer"]
df["performance_index"] = (df["selectivity"] * df["permeability_Barrer"]) / 1000.0

def quantile_bin(series, q_low=0.33, q_high=0.66):
    if series.isna().all(): return pd.Series(["NaN"] * len(series))
    low, high = series.quantile([q_low, q_high])
    bins = []
    for x in series:
        if pd.isna(x): bins.append("NaN")
        elif x <= low: bins.append("low")
        elif x <= high: bins.append("mid")
        else: bins.append("high")
    return pd.Series(bins)

if "temperature_C" in df.columns: df["temperature_bin"] = quantile_bin(df["temperature_C"])
if "filler_loading_wt%" in df.columns: df["load_bin"] = quantile_bin(df["filler_loading_wt%"])

# ===============================
# MISSING FLAGS
# ===============================
for c in num_cols:
    if c in df.columns:
        df[f"{c}_missing"] = df[c].isna().astype(int)

# ===============================
# FINAL SAVE
# ===============================
drop_cols = ["title","chunk_text"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df = df.fillna("NaN")

df.to_csv(OUT, index=False)
print(f"[✅] Stage-4 Final complete → {OUT}")
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
