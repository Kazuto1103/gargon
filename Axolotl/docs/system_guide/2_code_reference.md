# Referensi Kode - Fungsi & Struktur Komponen Runtime

Dokumen ini menjelaskan fungsi teknis, interface, dan behavior dari setiap skrip operasional di dalam direktori `src/core_pipeline/` dan `src/model_training/`. Gunakan dokumen ini sebagai panduan debugging, integration, dan extension.

---

## BAGIAN A: CORE PIPELINE (`src/core_pipeline/`)

### 1. `normalizer.py` (Gerbang Input & Sanitasi Teks)

Skrip ini bertindak sebagai pembersih hulu yang memastikan input teks siap diproses oleh model ML.

#### Tanggung Jawab Utama:
1. **Pemisahan Angka dan Satuan:** Memisahkan teks yang menempel menggunakan regular expression.
   - Input: `"10cm"`, `"2meter"`, `"90deg"`
   - Output: `"10 centimeter"`, `"2 meter"`, `"90 derajat"`

2. **Normalisasi Singkatan/Gaul Bahasa Indonesia:** Mengkonversi kata singkatan menjadi bentuk baku menggunakan dictionary `NORM_DICT`.
   - Input: `"mju"`, `"dpan"`, `"mnt"`, `"bsk"`, `"pagi"`
   - Output: `"maju"`, `"depan"`, `"menit"`, `"besok"`, `"pagi"`

3. **Penghapusan Kata Noise:** Mengeliminasi kata yang tidak membawa nilai semantik bagi robot.
   - Kata noise: `"woi"`, `"dong"`, `"heii"`, `"ya"`, `"deh"`, `"lah"`, `"nih"`
   - Contoh: `"Woi robot maju dong!"` → `"robot maju"` → (setelah final cleanup) → `"maju"`

#### Interface & Metode Krusial:

```python
# Main entry point
def normalize_text(text: str) -> str:
    """
    Normalisasi input teks secara keseluruhan.
    Input:  "mju dpan 10cm bsk pagi woi"
    Output: "maju depan 10 centimeter besok pagi"
    """
    pass

# Token extraction
def tokenize_sentence(text: str) -> List[str]:
    """
    Tokenize teks menjadi daftar token untuk CRF processing.
    Input:  "maju ke depan 2 meter"
    Output: ["maju", "ke", "depan", "2", "meter"]
    """
    pass

# Internal helper
NORM_DICT = {
    "mju": "maju",
    "dpan": "depan",
    "mnt": "menit",
    "bsk": "besok",
    ...
}

NOISE_WORDS = {"woi", "dong", "heii", "ya", "deh", "lah", "nih"}
```

#### Testing:
```python
assert normalize_text("mju dpan 10cm") == "maju depan 10 centimeter"
assert tokenize_sentence("maju ke depan") == ["maju", "ke", "depan"]
```

---

### 2. `grounding_translator.py` (Interpreter & Sistem Pertahanan Keamanan)

Skrip ini bertindak sebagai lapis keamanan (*fault-tolerant layer*) yang menerjemahkan prediksi ML menjadi instruksi robot tervalidasi.

#### Tanggung Jawab Utama:

1. **Intent Supremacy Rule:** Jika intent adalah `STOP_COMMAND`, abaikan semua kegagalan ekstraksi CRF dan langsung paksa output menjadi tindakan darurat.
   - Input: Intent=`STOP_COMMAND`, token_tags=`[("robot", "O"), ("berhenti", "B-ACTION"), ...]`
   - Output: `{"action": "STOP", "type": "EMERGENCY_BRAKE", ...}` (tidak peduli apakah parameter lain kosong)

2. **Token-Unit Binding:** Menggunakan taksonomi untuk memisahkan satuan spasial (`METER`, `DERAJAT`, `CENTIMETER`) dari satuan temporal (`MENIT`, `JAM`, `KALI`).
   - Tidak boleh ada kebetulan di mana `QUANTITY=2` dan `UNIT=JAM` disatukan ke parameter spasial robot.
   - Implementasi: Load `taxonomy.json`, validasi setiap mapping token→unit.

3. **Type Validation & Fallback:** Menangkap eror konversi data numerik.
   - Jika CRF salah label `"naik"` sebagai `B-QUANTITY`, sistem mendeteksi nilai non-numerik dan fallback ke `1.0`.
   - Jika `DIRECTION` slot kosong, assign default aman (misal: `FRONT` untuk `MOVE_FORWARD`).

4. **Timezone & Time Parsing:** Konversi relative time text (misal: `"besok jam 3 sore"`) menjadi ISO 8601 timestamp.
   - Menggunakan *reference time* sistem (hari/jam sekarang) untuk komputasi.

#### Interface Krusial:

```python
class GroundingEngine:
    def __init__(self, taxonomy_path: str = "taxonomy.json"):
        """Load taksonomi mapping dari file."""
        self.taxonomy = load_taxonomy(taxonomy_path)
        self.spatial_units = self.taxonomy["spatial_units"]
        self.temporal_units = self.taxonomy["temporal_units"]
        self.action_map = self.taxonomy["action_grounding"]
        ...

    def translate(self, intent: str, token_tag_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Terjemahkan prediksi ML menjadi JSON command payload.
        
        Input:
            intent = "DIRECT_COMMAND"
            token_tag_pairs = [("maju", "B-ACTION"), ("ke", "O"), ("depan", "B-DIRECTION"), ("2", "B-QUANTITY"), ("meter", "B-UNIT")]
        
        Output:
            {
                "status": "SUCCESS",
                "command": {"intent": "DIRECT_COMMAND", "action": "MOVE_FORWARD"},
                "parameters": {
                    "spatial": {"direction": "FRONT", "quantity": 2.0, "unit": "METER"},
                    "temporal": {"is_scheduled": false, ...}
                },
                "pipeline_metadata": {"fallback_triggered": false, ...}
            }
        """
        # Step 1: Check Intent Supremacy
        if intent == "STOP_COMMAND":
            return self._generate_stop_command()
        
        # Step 2: Extract slots from token_tag_pairs
        slots = self._extract_slots(token_tag_pairs)
        
        # Step 3: Validate & bind slots to taxonomy
        validated_slots = self._validate_and_bind(slots)
        
        # Step 4: Generate final JSON
        return self._build_json_payload(intent, validated_slots)
```

#### Aturan-Aturan Tegas (Hard Rules):

```python
# Rule 1: Intent Supremacy
if intent == "STOP_COMMAND":
    return {
        "status": "SUCCESS",
        "command": {"intent": "STOP_COMMAND", "action": "STOP", "type": "EMERGENCY_BRAKE"},
        "parameters": {"spatial": {...}, "temporal": {...}},  # Filled with defaults
        "pipeline_metadata": {"fallback_triggered": True, "fallback_reason": "INTENT_SUPREMACY: STOP_COMMAND overrides all slots"}
    }

# Rule 2: Direction defaults per action
DIRECTION_DEFAULTS = {
    "MOVE_FORWARD": "FRONT",
    "MOVE_BACKWARD": "BACK",
    "MOVE_ARM_UP": "UP",
    "MOVE_ARM_DOWN": "DOWN",
    ...
}

# Rule 3: Type validation
def validate_quantity(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 1.0  # Safe default
    return float(value) if value else 1.0

# Rule 4: Spatial-Temporal unit separation
SPATIAL_UNITS = {"METER", "CENTIMETER", "KILOMETER", "DERAJAT"}
TEMPORAL_UNITS = {"MENIT", "JAM", "HARI", "KALI"}
```

---

### 3. `e2e_simulation.py` (Orchestrator Pipeline Akhir & Test Harness)

Skrip ini mengintegrasikan semua komponen (normalizer, ML models, grounding engine) menjadi satu sistem yang berfungsi end-to-end.

#### Tanggung Jawab Utama:

1. **Model Loading:** Memuat semua artifact biner dari folder `build/`.
   - `vectorizer.pkl`: TF-IDF vocabulary
   - `intent_model.pkl`: Trained SVM classifier
   - `slot_model.pkl`: Trained CRF model

2. **Pipeline Orchestration:** Mengkoordinasikan aliran data dari input teks hingga output JSON.
   - Step 1: `normalize_text(raw_input)` → cleaned text
   - Step 2: Tokenize & extract SVM features → intent prediction
   - Step 3: Tokenize & extract CRF features → slot prediction
   - Step 4: `grounding_engine.translate(intent, slots)` → final JSON

3. **Test Harness:** Menyediakan automated test suite dan interactive REPL.
   - Automated: 12 test cases coverage untuk intent types, edge cases, fallbacks.
   - Interactive: CLI loop untuk manual testing real-time.

4. **Logging & Metrics:** Mencatat execution trace untuk debugging.
   - Log file: `src/core_pipeline/log/e2e_simulation.log`
   - Output: JSON payload + metadata per test case

#### Class Diagram:

```python
class NLUPipeline:
    def __init__(self, build_dir: str = "build"):
        """
        Inisialisasi pipeline dengan memuat semua model.
        Raises FileNotFoundError jika ada model yang hilang.
        """
        self.vectorizer = load_pickle(os.path.join(build_dir, "vectorizer.pkl"))
        self.svm_model = load_pickle(os.path.join(build_dir, "intent_model.pkl"))
        self.crf_model = load_pickle(os.path.join(build_dir, "slot_model.pkl"))
        self.grounding_engine = GroundingEngine()

    def predict_and_ground(self, raw_text: str) -> Dict[str, Any]:
        """
        Pipeline lengkap: normalize → predict intent → predict slots → ground.
        
        Args:
            raw_text: Input teks dari user (misal: "maju ke depan 2 meter")
        
        Returns:
            JSON command payload siap dikonsumsi robot controller.
        """
        # Step 1: Normalize
        normalized = normalize_text(raw_text)
        
        # Step 2: Intent classification (SVM)
        clean_text = preprocess_text(raw_text)
        vectorized = self.vectorizer.transform([clean_text])
        intent = self.svm_model.predict(vectorized)[0]
        
        # Step 3: Slot extraction (CRF)
        tokens = tokenize_sentence(normalized)
        features = sent2features(tokens)
        slot_tags = self.crf_model.predict([features])[0]
        token_tag_pairs = list(zip(tokens, slot_tags))
        
        # Step 4: Grounding
        payload = self.grounding_engine.translate(intent, token_tag_pairs)
        
        return payload
```

#### Entry Point:

```python
def main():
    # Setup logging
    log_dir = "src/core_pipeline/log"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize pipeline
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_dir = os.path.join(workspace_root, "build")
    pipeline = NLUPipeline(build_dir=build_dir)
    
    # Run tests
    if "--interactive" in sys.argv:
        run_interactive_repl(pipeline)
    else:
        run_automated_tests(pipeline)

if __name__ == "__main__":
    main()
```

---

## BAGIAN B: MODEL TRAINING (`src/model_training/`)

### 1. `preprocess_features.py` (Feature Extraction)

Skrip ini mengekstrak representasi numerik dari teks untuk kedua model ML.

#### Fungsi Utama:

```python
def preprocess_text(text: str) -> str:
    """
    Preprocessing khusus untuk SVM: lowercase, remove special chars, etc.
    Input:  "MAJU ke DEPAN 2 METER"
    Output: "maju ke depan 2 meter"
    """
    pass

def sent2features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Ekstrak fitur kontekstual per token untuk CRF.
    Input:  ["maju", "ke", "depan", "2", "meter"]
    Output: [
        {"word": "maju", "BOS": True, "word[-3:]": "aju", "pos": "VB", ...},
        {"word": "ke", "word[-3:]": "ke", "pos": "IN", ...},
        ...
    ]
    """
    pass
```

---

### 2. `train_models.py` (Model Training)

Skrip untuk melatih SVM dan CRF dari dataset yang sudah berlabel.

#### Output:
- `build/vectorizer.pkl`
- `build/intent_model.pkl`
- `build/slot_model.pkl`
- Metrics: Precision, Recall, F1-Score per intent/slot

---

### 3. `verify_models.py` & `verify_preprocessed.py` (Validation Scripts)

Skrip untuk validasi kualitas preprocessing dan model performance.
