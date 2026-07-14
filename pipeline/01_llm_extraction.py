import json
import os
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ===============================
# CONFIG
# ===============================

# pipeline/ klasörünün bir üstü repository ana klasörüdür.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHUNK_DIR = OUTPUTS_DIR / "chunks"
OUTPUT_DIR = OUTPUTS_DIR / "llm_extraction"
META_CSV = PROJECT_ROOT / "results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
SLEEP = 2

# Repository ana klasöründeki .env dosyasını yükle.
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY bulunamadı. Repository ana klasöründe bir .env "
        "dosyası oluşturun ve OPENAI_API_KEY değerini ekleyin."
    )

client = OpenAI(api_key=api_key)

LOG_FILE = OUTPUT_DIR / "debug_log.txt"

def log(message: str):
    """Debug mesajlarını hem terminale hem dosyaya yaz."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ===============================
# CHUNK’TAN BAŞLIK ÇIKARMA FONKSİYONU
# ===============================
def extract_title_from_chunk(fpath):
    """Chunk dosyasının ilk satırından makale başlığını tahmin eder."""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            first_line = f.readline()
        data = json.loads(first_line)
        text = data.get("text", "")
        if not text:
            return None

        # 'abstract' kelimesinden öncesi genelde başlık kısmıdır
        if "abstract" in text.lower():
            before_abs = text.split("abstract")[0]
        else:
            before_abs = text[:400]

        candidate = before_abs.strip().split("\n")[0]
        title = candidate.strip()
        if 10 < len(title) < 200:
            return title
        return None
    except Exception:
        return None


# ===============================
# PROMPT (v4 – Unit-Aware & Anti-Bias)
# ===============================
LLM_PROMPT = """
You are an expert in polymer science and mixed matrix membranes (MMM).
Your task is to extract structured scientific information from the provided text.

### Output Format

Return a valid **JSON array**.
Each element of the array represents one distinct experimental condition and must include the following **15 fields**:

[
"doi", "title", "polymer", "filler", "filler_modification", "polymer_modification",
"gas_type", "application_type", "temperature_C", "pressure_bar",
"filler_loading_wt%", "permeability_Barrer", "selectivity", "compatibility", "notes"
]

---

### Field Guidelines

* **doi**: article DOI if available, else null  
* **title**: paper or study title, if mentioned; else null  
* **polymer**: main polymer matrix  
* **filler**: filler or additive name  
* **filler_modification**: surface treatment or functionalization, else null  
* **polymer_modification**: chemical or structural modification of polymer, else null  
* **gas_type**: gas or vapor tested (CO₂, CH₄, N₂, H₂O, etc.)  
* **application_type**: overall application (gas separation, pervaporation, desalination, proton exchange, etc.)  
* **temperature_C**:  
  * Convert to degrees Celsius (°C).  
  * If reported in Kelvin (K), subtract 273.15 (e.g., 298 K → 25 °C).  
  * Always return numeric float or null.  
* **pressure_bar**:  
  * Convert to bar if in other units.  
  * 1 atm ≈ 1.013 bar, 1 kPa = 0.01 bar, 1 MPa = 10 bar.  
* **filler_loading_wt%**:  
  * Report as numeric percentage (e.g., 10 for 10 wt%).  
  * If expressed as fraction or phr, convert to equivalent wt%.  
* **permeability_Barrer**: permeability in Barrer; if in other units, convert where possible.  
* **selectivity**: selectivity ratio (numeric).  
* **compatibility**:  
  * "Compatible" → strong interfacial adhesion, uniform dispersion, no agglomeration.  
  * "Non-Compatible" → agglomeration, voids, phase separation, weak adhesion.  
  * "Not Reported" → no explicit statement about polymer–filler interface.  
* **notes**: one short sentence summarizing the main performance observation or experimental finding.

---

### Formatting Rules

* Always return a syntactically valid **JSON array**, even if there is only one experiment.  
* Use `null` for missing or unreported values.  
* Convert all units according to the above rules.  
* All numeric values must be plain numbers (no units or symbols).  
* If multiple experimental loadings or conditions are described, include multiple JSON objects (one per condition).

---

### Important

* Do **not** copy chemical names or units from this prompt.  
* Infer all information **only** from the provided text.  
* Ensure the JSON is valid and strictly follows the defined 15-field schema.

### Text to analyze:

{chunk_text}
"""

# ===============================
# EXTRACTION LOOP
# ===============================
def extract_from_chunk(text: str, file_idx: int, total_files: int, chunk_idx: int, total_chunks: int):
    """Tek bir metin chunk'ından JSON döndürür."""
    for attempt in range(MAX_RETRIES):
        try:
            log(f"→ [File {file_idx}/{total_files}] [Chunk {chunk_idx}/{total_chunks}] Extracting (attempt {attempt+1}) | {len(text)} chars")

            prompt_filled = LLM_PROMPT.replace("{chunk_text}", text)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a scientific data extractor."},
                    {"role": "user", "content": prompt_filled},
                ],
                temperature=0,
            )

            content = response.choices[0].message.content.strip()

            if not content:
                raise ValueError("Empty response from model")

            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.startswith("```"):
                content = content[len("```"):].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                log(f"[WARN] [File {file_idx}/{total_files}] [Chunk {chunk_idx}/{total_chunks}] Invalid JSON response, first 200 chars:\n{content[:200]}...")
                raise

            log(f"✓ [File {file_idx}/{total_files}] [Chunk {chunk_idx}/{total_chunks}] Extraction success ({len(data)} items)")
            return data

        except Exception as e:
            log(f"[WARN] [File {file_idx}/{total_files}] [Chunk {chunk_idx}/{total_chunks}] Attempt {attempt+1} failed: {e}")
            log(traceback.format_exc())
            time.sleep(SLEEP)

    log(f"[ERROR] [File {file_idx}/{total_files}] [Chunk {chunk_idx}/{total_chunks}] All retries failed.")
    return []


def main():
    META_CSV = BASE_DIR / "results.csv"    
    meta = pd.read_csv(META_CSV)
    meta = meta.dropna(subset=["doi", "title"])
    meta["doi_clean"] = (
        meta["doi"]
        .astype(str)
        .str.strip()
        .str.replace("https://doi.org/", "", regex=False)
        .str.replace("https://dx.doi.org/", "", regex=False)
        .str.replace("doi:", "", regex=False)
        .str.replace("/", "_", regex=False)
    )
    doi_to_title = meta.set_index("doi_clean")["title"].to_dict()
    log(f"[INFO] DOI-title mapping loaded ({len(doi_to_title)} pairs).")

    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".jsonl")]
    total_files = len(files)
    log(f"[INFO] Found {total_files} chunk files.")

    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    processed = set()

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                processed = set(json.load(f))
            log(f"[INFO] Loaded checkpoint: {len(processed)} files already processed.")
        except Exception as e:
            log(f"[WARN] Could not load checkpoint: {e}")

    for file_idx, fname in enumerate(files, start=1):
        if fname in processed:
            log(f"[SKIP] [File {file_idx}/{total_files}] {fname} already processed.")
            continue

        doi_key = fname.replace(".jsonl", "")
        fpath = CHUNK_DIR / fname
        out_path = OUTPUT_DIR / f"{doi_key}_extracted.json"

        title_from_csv = doi_to_title.get(doi_key)
        if not title_from_csv:
            title_from_csv = extract_title_from_chunk(fpath)
            if title_from_csv:
                log(f"[DEBUG] Auto-detected title from chunk: {title_from_csv[:100]}")

        if os.path.exists(out_path):
            processed.add(fname)
            continue

        log(f"\n[INFO] [File {file_idx}/{total_files}] Processing {fname}")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]

            # 🔒 DOI’yi sabitle (ilk chunk’tan veya dosya adından)
            doi_fixed = doi_key.replace("_", "/")

            for chunk in lines:
                if "doi" not in chunk or not chunk.get("doi"):
                    chunk["doi"] = doi_fixed
                if "title" not in chunk or not chunk.get("title"):
                    chunk["title"] = title_from_csv or None

            total_chunks = len(lines)
            extracted = []

            for chunk_idx, chunk in enumerate(lines, start=1):
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                result = extract_from_chunk(text, file_idx, total_files, chunk_idx, total_chunks)

                if isinstance(result, list):
                    for item in result:
                        # 🔒 DOI sabit tut
                        item["doi"] = doi_fixed
                        # Eksik başlığı tamamla
                        if not item.get("title"):
                            item["title"] = title_from_csv or None
                        extracted.append(item)

            if extracted:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(extracted, f, indent=2, ensure_ascii=False)
                log(f"[DONE] Saved → {out_path}")

                try:
                    csv_path = OUTPUT_DIR / "results_llm_extracted.csv"
                    df = pd.DataFrame(extracted)
                    if os.path.exists(csv_path):
                        df.to_csv(csv_path, mode="a", index=False, header=False)
                    else:
                        df.to_csv(csv_path, index=False)
                    log(f"[CSV] Appended {len(df)} rows → {csv_path}")
                except Exception as e:
                    log(f"[WARN] Could not save to CSV: {e}")

            else:
                log(f"[FAIL] No data extracted from {fname}")

            processed.add(fname)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(list(processed), f, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"[ERROR] Unexpected error while processing {fname}: {e}")
            log(traceback.format_exc())

    log("[ALL DONE] All files processed successfully.")


if __name__ == "__main__":
    main()
