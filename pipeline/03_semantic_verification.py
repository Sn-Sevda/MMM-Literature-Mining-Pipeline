import json
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REFINED_CSV = (
    OUTPUTS_DIR
    / "stage1p5_refined"
    / "results_stage1p5_refined.csv"
)

CHUNK_DIR = OUTPUTS_DIR / "chunks"
VERIFIED_DIR = OUTPUTS_DIR / "verified"

SUMMARY_CSV = VERIFIED_DIR / "verification_summary.csv"
LOG_FILE = VERIFIED_DIR / "debug_log.txt"
CHECKPOINT_FILE = VERIFIED_DIR / "checkpoint.txt"

MODEL = "claude-3-5-haiku-20241022"
MAX_RETRIES = 3
SLEEP = 3

VERIFIED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY bulunamadı. Repository ana klasöründe "
        "bir .env dosyası oluşturun ve ANTHROPIC_API_KEY değerini ekleyin."
    )

client = anthropic.Anthropic(api_key=anthropic_api_key)

# ===============================
#  CHUNK TEXT EKLEME
# ===============================
print(f"🔍 Checking for missing text fields in: {REFINED_CSV}")
df = pd.read_csv(REFINED_CSV)

if "chunk_text" not in df.columns:
    print("⚙️  No 'chunk_text' column found — recovering from chunk files...")
    chunk_map = {}

    for fname in tqdm(os.listdir(CHUNK_DIR), desc="Scanning chunk files"):
        if not fname.endswith(".jsonl"):
            continue
        doi = fname.replace(".jsonl", "").replace("_", "/")
        try:
            with open(os.path.join(CHUNK_DIR, fname), "r", encoding="utf-8") as f:
                texts = [json.loads(line).get("text", "") for line in f]
                texts = [t.strip() for t in texts if t.strip()]
                if texts:
                    chunk_map[doi] = " ".join(texts)[:5000]  # 5000 karakter sınırı
        except Exception:
            continue

    print(f" Collected text for {len(chunk_map)} DOIs")

    df["doi_clean"] = df["doi"].astype(str).str.strip().str.replace("https://doi.org/", "").str.replace("doi:", "")
    df["chunk_text"] = df["doi_clean"].map(chunk_map)
    df.drop(columns=["doi_clean"], inplace=True)

    new_csv = (
        OUTPUTS_DIR
        / "stage1p5_refined"
        / "results_stage1p5_with_chunks.csv"
    )
    df.to_csv(new_csv, index=False)
    print(f" Updated CSV saved → {new_csv}")
    CSV_PATH = new_csv
else:
    CSV_PATH = REFINED_CSV
    print(" 'chunk_text' column already exists, using refined CSV directly.")

# ===============================
#  DOĞRULAMA
# ===============================
FIELD_WEIGHTS = {
    "polymer": 0.15,
    "filler": 0.15,
    "compatibility": 0.20,
    "gas_type": 0.10,
    "temperature_C": 0.10,
    "pressure_bar": 0.10,
    "permeability_Barrer": 0.10,
    "selectivity": 0.10,
}

def weighted_score(results):
    s = 0
    for r in results:
        s += r.get("confidence", 0) * FIELD_WEIGHTS.get(r["field"], 0)
    return round(s, 3)

def classify_score(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.5:
        return "Medium"
    return "Low"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_checkpoint(doi):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(doi + "\n")

print(f" Reading refined data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
if "doi" not in df.columns:
    raise ValueError("CSV dosyasında 'doi' sütunu bulunamadı.")

verified_dois = load_checkpoint()
all_dois = df["doi"].dropna().unique()

for doi in tqdm(all_dois, desc="Semantic verification"):
    if doi in verified_dois:
        continue

    subset = df[df["doi"] == doi]
    first_row = subset.iloc[0].to_dict()
    input_fields = {k: first_row.get(k, "") for k in FIELD_WEIGHTS.keys() if k in first_row}

    # Birden fazla metin sütununu birleştir
    possible_text_fields = ["notes", "chunk_text", "paragraph_text", "context"]
    texts = [
        str(first_row.get(col, "")).strip()
        for col in possible_text_fields
        if str(first_row.get(col, "")).lower() not in ["nan", "none", ""]
    ]
    source_text = "\n\n".join(texts).strip()
    if not source_text:
        source_text = "(No source text available)"

    system_prompt = """
You are an expert in polymer science verifying extracted experimental data.
Compare each JSON field with the source text.
Mark 'verified': true/false and a confidence score (0–1).
Accept ±5% tolerance for numeric values and equivalent expressions (e.g. CO2 ≈ carbon dioxide).
Return ONLY a JSON array like:
[{"field":"polymer","verified":true,"confidence":0.93}, ...]
Do not include any explanation or commentary. Respond with a valid JSON array only.
"""

    user_message = f"Source text:\n{source_text}\n\nExtracted fields:\n{json.dumps(input_fields, indent=2)}"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=600,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_output = response.content[0].text.strip()
            start_idx = raw_output.find("[")
            end_idx = raw_output.rfind("]")
            if start_idx != -1 and end_idx != -1:
                json_text = raw_output[start_idx:end_idx + 1]
            else:
                json_text = raw_output

            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"[WARN] {doi} JSON parsing failed ({e}) → saving raw output.")
                safe_doi = doi.replace("/", "_")
                with open(os.path.join(VERIFIED_DIR, f"{safe_doi}_raw.txt"), "w") as f:
                    f.write(raw_output)
                continue
            break

        except Exception as e:
            print(f"[WARN] {doi} attempt {attempt+1} failed → {type(e).__name__}: {e}")
            time.sleep(SLEEP)
    else:
        print(f"[ERROR] {doi} verification failed after retries.")
        continue

    score = weighted_score(result)
    level = classify_score(score)
    true_c = sum(1 for r in result if r["verified"])
    false_c = sum(1 for r in result if not r["verified"])
    avg_conf = round(sum(r["confidence"] for r in result) / len(result), 3)

    summary = {
        "doi": doi,
        "true_count": true_c,
        "false_count": false_c,
        "avg_confidence": avg_conf,
        "weighted_score": score,
        "verification_level": level,
        "timestamp": datetime.now().isoformat(),
    }

    safe_doi = doi.replace("/", "_")
    verified_path = os.path.join(VERIFIED_DIR, f"{safe_doi}_verified.json")
    with open(verified_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(SUMMARY_CSV, "a") as f:
        if f.tell() == 0:
            f.write("doi,true_count,false_count,avg_confidence,weighted_score,verification_level,timestamp\n")
        f.write(",".join(map(str, summary.values())) + "\n")

    save_checkpoint(doi)
    with open(LOG_FILE, "a") as log:
        log.write(f"[{datetime.now()}] {doi} → {level} (score={score})\n")

print("\n Verification completed. Results saved in outputs/verified/")
