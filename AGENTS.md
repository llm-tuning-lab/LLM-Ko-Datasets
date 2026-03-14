# LLM-Ko-Datasets — AGENTS.md

**Generated:** 2026-03-14  
**Project:** Korean LLM dataset curation guide  
**Type:** Public fork (reference documentation)

---

## OVERVIEW

Comprehensive guide to Korean + English + multilingual datasets for LLM training. Covers Pre-training, Mid-training, Post-training (SFT/DPO/RLHF/GRPO), and evaluation datasets. Includes free translation strategies and data pipeline recommendations.

**Upstream:** https://github.com/gyunggyung/LLM-Ko-Datasets  
**Maintainer:** @gyunggyung (Kiwoong Yeom)

---

## STRUCTURE

```
LLM-Ko-Datasets/
  README.md              # Main dataset catalog (Korean, 551 lines)
  LICENSE                # License information
```

**Total:** 2 files (documentation only, no code)

---

## USAGE

### Browse Dataset Catalog

```bash
# View full catalog
cat README.md

# Search by category
grep -A 10 "## Pre-training" README.md
grep -A 10 "## Post-training" README.md
grep -A 10 "## 한국어" README.md

# Find specific datasets
grep "FineWeb" README.md
grep "KoCommercial" README.md
grep "GRPO" README.md
```

### Dataset Categories

| Category | Count | Key Datasets |
|----------|-------|--------------|
| **Pre-training (English)** | 20+ | FineWeb, RedPajama-V2, Dolma, SmolLM-Corpus |
| **Pre-training (Korean)** | 15+ | Korean Wikipedia, WanJuan-Korean, KORMo datasets |
| **Mid-training** | 10+ | Dolma3 Dolmino, Cosmopedia-ko-synth |
| **SFT** | 30+ | KoCommercial-Dataset (1.44M), koVast (685K) |
| **DPO/Preference** | 10+ | ko_Ultrafeedback_binarized, K2-Feedback |
| **GRPO/RL** | 8+ | NuminaMath-TIR, Dolci-RL-Zero-Math |
| **Evaluation** | 6+ | KMMLU, HAE-RAE-BENCH, LogicKor |

---

## KEY RESOURCES

### Free Translation Strategy

**Tools:**
- Google Translate (비공식): Unlimited, highest Korean quality
- DeepL API Free: 500K chars/month
- NLLB (Meta): Unlimited, 200 languages
- LFM2-1.2B-KoEn-MT: Unlimited, SOTA 1.2B translation

**Pipeline:**
```
English dataset → Google Translate / LFM2 → Quality filter → Korean dataset
```

### Recommended Data Pipelines

**Pre-training (300B tokens):**
- English 50%: FineWeb-Edu, SmolLM-Corpus, The Stack
- Korean 50%: Korean Wikipedia, korean_textbooks, KOREAN-SyntheticText

**Mid-training (50-100B tokens):**
- Dolma3 Dolmino Mix, Korean Pretraining Collection, Synthetic CoT

**SFT:**
- 1st: KoCommercial-Dataset (1.44M, commercial-friendly)
- 2nd: open-korean-instructions
- 3rd: Magpie-Pro-MT-300K-ko (synthetic)

**GRPO (Alignment):**
- Stage 1 DPO: ko_Ultrafeedback_binarized
- Stage 2 GRPO: NuminaMath-CoT-Ko, NuminaMath-TIR

---

## NOTES

- **Fork status:** Reference documentation, no code
- **Update frequency:** Last updated 2026-01-12
- **Language:** Korean (primary), English (technical terms)
- **HuggingFace-first:** All datasets prioritize HuggingFace availability
- **Related projects:** 
  - LFM2-KoEn-Tuning (translation model)
  - KORMo-Team datasets (largest Korean collection)
  - Solar Open, K-EXAONE, A.X K1 (Korean LLM tech reports)

---

**Last Updated:** 2026-01-12  
**License:** Various (check individual datasets)
