# Workflow CI - Training Model

Repository ini berisi pipeline Continuous Integration (CI) untuk melatih model machine learning menggunakan MLflow.

## Struktur Repository

- `.github/workflows/training.yml` → Workflow GitHub Actions untuk menjalankan training otomatis
- `MLProject/` → Folder MLflow Project yang berisi:
  - modelling.py
  - MLproject
  - conda.yaml
  - dataset_preprocessing

Workflow akan otomatis menjalankan training model setiap ada push ke repository.