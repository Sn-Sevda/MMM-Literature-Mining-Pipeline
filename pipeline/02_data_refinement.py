import os
import pandas as pd
import numpy as np
import time

# ===============================
# CONFIG
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_CSV = (
    OUTPUTS_DIR
    / "llm_extraction"
    / "results_llm_extracted.csv"
)

OUTPUT_DIR = OUTPUTS_DIR / "stage1p5_refined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFINED_CSV = OUTPUT_DIR / "results_stage1p5_refined.csv"
DROPPED_CSV = OUTPUT_DIR / "results_stage1p5_dropped.csv"
LOG_FILE = OUTPUT_DIR / "refinement_log.txt"

# ===============================
# HELPERS
# ===============================
def log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def calc_info_score(row):
    """Veri anlamlılık skorunu hesapla (0–1 arası)"""
    filled = row.notna().sum()
    total = len(row)
    fill_ratio = filled / total

    notes_len = len(str(row.get("notes", "")))
    notes_score = min(notes_len / 150, 1.0)

    comp_score = 1.0 if str(row.get("compatibility", "")).strip().lower() not in ["", "none", "nan"] else 0.0

    return round((fill_ratio * 0.7 + notes_score * 0.2 + comp_score * 0.1), 3)

# ===============================
# MAIN PIPELINE
# ===============================
def main():
    log("=== Stage 1.5 Refinement Started ===")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Stage 1 çıktı dosyası bulunamadı: {INPUT_CSV}\n"
            "Önce pipeline/01_llm_extraction.py dosyasını çalıştırın."
        )

    # Veri yükleme
    df = pd.read_csv(INPUT_CSV)
    log(f"[INFO] Loaded {len(df):,} rows from Stage 1.")

    #  Çok eksik satırları at (örneğin %60’tan fazlası boş)
    threshold = int(df.shape[1] * 0.4)
    df = df.dropna(thresh=threshold)
    log(f"[CLEAN] Dropped rows with >60% missing values. Remaining: {len(df):,}")

    #  Yinelenen satırları temizle
    dedup_cols = ["doi", "polymer", "filler", "filler_loading_wt%", 
                  "gas_type", "temperature_C", "pressure_bar", "compatibility"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    dropped_duplicates = before - len(df)
    log(f"[CLEAN] Removed {dropped_duplicates:,} duplicate rows.")

    #  Anlamlılık skoru ekle
    df["info_score"] = df.apply(calc_info_score, axis=1)
    log("[INFO] Info score added for all rows.")

    #  En anlamlı %50 veriyi seç
    cutoff = df["info_score"].quantile(0.5)
    refined_df = df[df["info_score"] >= cutoff].copy()
    dropped_df = df[df["info_score"] < cutoff].copy()
    log(f"[FILTER] Selected top 50% most informative rows. {len(refined_df):,} kept, {len(dropped_df):,} dropped.")

    #  Tüm sütunlarda boşlukları 'Unknown' ile doldur
    refined_before_unknowns = (refined_df.isna() | refined_df.eq("") | refined_df.eq(" ")).sum().sum()
    
    refined_df = refined_df.replace(["", " ", "nan", "NaN", "None", "null", np.nan], "Unknown")
    refined_df = refined_df.fillna("Unknown")

    refined_after_unknowns = (refined_df == "Unknown").sum().sum()
    newly_filled = refined_after_unknowns - refined_before_unknowns
    log(f"[FILL] All missing/empty cells filled with 'Unknown'. ({newly_filled:,} fields updated)")

    #  Çıktıları kaydet
    refined_df.to_csv(REFINED_CSV, index=False)
    dropped_df.to_csv(DROPPED_CSV, index=False)
    log(f"[SAVE] Refined data → {REFINED_CSV}")
    log(f"[SAVE] Dropped data → {DROPPED_CSV}")

    #  Genel özet
    avg_score = refined_df["info_score"].mean()
    comp_ratio = (refined_df["compatibility"].ne("Unknown").sum() / len(refined_df)) * 100
    log(f"[SUMMARY] Avg info_score: {avg_score:.3f}")
    log(f"[SUMMARY] Compatibility completeness: {comp_ratio:.1f}%")

    log("=== Stage 1.5 Refinement Completed Successfully ===")

if __name__ == "__main__":
    main()
