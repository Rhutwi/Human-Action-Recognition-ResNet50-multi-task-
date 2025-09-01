# Human Action Recognition — Multi-Task ResNet50

**One model, two heads:** predict **action** and **person presence** together with a custom multi-task loss. Designed to be **lightweight** and **Colab-friendly**.

## ✨ Features
- Transfer-learning with **ResNet50** backbone
- **Multi-task head**: (A) action class, (B) person present (yes/no)
- Custom **weighted loss** to balance tasks
- Simple dataset interface (drop your images + CSV)

## 🧰 Tech Stack
PyTorch · torchvision · numpy · pandas · scikit-learn

## 🚀 Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python train.py --data_root data/ --labels_csv data/train_labels.csv --epochs 10
