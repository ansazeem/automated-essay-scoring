# Automated Essay Scoring

Fine-tuned RoBERTa for automated essay scoring on the ASAP 2.0 dataset, with multilingual extension to German and Spanish.

## Notebooks

| Notebook                                   | Description                                         |
| ------------------------------------------ | --------------------------------------------------- |
| `01_ASAP_Preprocessing_EDA_Baseline.ipynb` | EDA, TF-IDF preprocessing, XGBoost baseline         |
| `aes_roberta_english.ipynb`                | RoBERTa fine-tuning on English ASAP 2.0             |
| `aes_spanish_german_monolingual.ipynb`     | Monolingual models for Spanish and German           |
| `aes_crosslingual_de_to_es.ipynb`          | Zero-shot cross-lingual transfer (German → Spanish) |

## Results

| Model                        | QWK   | Pearson | MAE   | RMSE  |
| ---------------------------- | ----- | ------- | ----- | ----- |
| TF-IDF + XGBoost (English)   | 0.677 | 0.693   | 0.480 | 0.753 |
| RoBERTa (English)            | 0.777 | 0.832   | 0.486 | 0.634 |
| Spanish (monolingual)        | 0.562 | 0.639   | 0.735 | 0.957 |
| German (monolingual)         | 0.794 | 0.850   | 0.415 | 0.535 |
| German → Spanish (zero-shot) | 0.351 | 0.507   | 0.824 | 1.144 |

## Dataset

- English: [ASAP 2.0 on Kaggle](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)
- Spanish & German: included in repo (`aes_spanish.tsv`, `german_raw/`)

## Approach

- Baseline: TF-IDF + XGBoost
- Final: RoBERTa-base fine-tuned as regression model (QWK-optimized)
- Regression chosen over classification to exploit ordinal score structure
- Extended to multilingual and zero-shot cross-lingual settings

## Setup

```bash
pip install torch transformers datasets scikit-learn xgboost scipy
```
