# Walkthrough - Phase 5b: Grounding Translator Hotfix (Token-Unit Binding)

Sesi ini berfokus pada perbaikan kebocoran logika semantik (*semantic logic leak*) pada [grounding_translator.py](file:///d:/Project/Axolotl/grounding_translator.py) untuk menangani kalimat multi-slot (seperti perintah berulang dengan interval waktu dan kuantitas gerakan spasial).

---

## Masalah & Perbaikan Kebocoran Logika

### Masalah Awal (The Bug)
Pada kalimat berulang seperti *"tiap 5 menit belok ke kanan 90 derajat"*, nilai slot kuantitas spasial (`90` dan `derajat`) tertimpa oleh parameter temporal (`5` dan `menit`), sehingga parameter spasial kehilangan datanya dan hanya menyisakan data temporal di slot spasial.

### Solusi: Deterministic Token-Unit Binding (Phase 5b)
1. **Taksonomi Unit yang Ketat (Strict Taxonomy)**:
   * `SPATIAL_UNITS = ['METER', 'CENTIMETER', 'DERAJAT', 'STEP']`
   * `TEMPORAL_UNITS = ['MENIT', 'JAM', 'DETIK', 'KALI']`
2. **Sequential Binding Logic**:
   * Menelusuri daftar token-tag secara linier.
   * Ketika menemukan token `B-QUANTITY` atau `I-QUANTITY`, pencarian diarahkan ke depan (*look-ahead*) untuk mencari token `B-UNIT` atau `I-UNIT` terdekat.
   * Kuantitas yang diparsing kemudian diarahkan ke parameter spasial atau temporal bergantung pada taksonomi unit standar yang dicocokkan.

---

## Perubahan Skema JSON

Skema parameter diperbarui untuk mendukung pemisahan spasial dan interval temporal secara tegas:
```json
{
  "parameters": {
    "spatial": {
      "direction": "str or null",
      "quantity": "float or null",
      "unit": "str or null"
    },
    "temporal": {
      "is_scheduled": "bool",
      "execute_at": "str or null",
      "interval_quantity": "float or null",
      "interval_unit": "str or null"
    }
  }
}
```

---

## Verifikasi Hasil Hotfix (Case 8)

Hasil penanganan skenario baru *"tiap 5 menit belok ke kanan 90 derajat"* diuji menggunakan unit test case pada verification harness:

```json
// Case 8 – Multi-slot REPEATED_COMMAND: 'tiap 5 menit belok ke kanan 90 derajat'
{
  "status": "SUCCESS",
  "command": {
    "intent": "REPEATED_COMMAND",
    "action": "ROTATE"
  },
  "parameters": {
    "spatial": {
      "direction": "RIGHT",
      "quantity": 90.0,
      "unit": "DERAJAT"
    },
    "temporal": {
      "is_scheduled": true,
      "execute_at": null,
      "interval_quantity": 5.0,
      "interval_unit": "MENIT"
    }
  },
  "pipeline_metadata": {
    "fallback_triggered": false,
    "fallback_reason": null
  }
}
```

---

## File Log yang Diperbarui
* [log/grounding_translator.log](file:///d:/Project/Axolotl/log/grounding_translator.log) – Log tes standalone modul grounding dengan Case 8 baru.
* [log/e2e_simulation.log](file:///d:/Project/Axolotl/log/e2e_simulation.log) – Log simulasi NLU terintegrasi penuh.
