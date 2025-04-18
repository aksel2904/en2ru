# EN→RU Neural Translator (RNN + Attention)
---

## Архитектура проекта

- **Seq2Seq на RNN + Attention**
- Реализация на **PyTorch Lightning**
- **BPE-токенизация (SentencePiece)**
- **DVC** для хранения:
  - **всех датасетов** (`data/`)
  - **весов модели** (`weights/*.ckpt`)
  - **предсказаний и метрик** (`outputs/`)
- **Hydra** для управления конфигурацией
- **BLEU**-score для оценки качества перевода

---
# Настройка Google Drive как DVC remote

## Что находится в DVC?

Репозиторий использует [DVC](https://dvc.org/) с подключением к **Google Drive** как к remote-хранилищу.

Хранятся:
- `data/` — все используемые параллельные корпуса
- `weights/` — сохранённые чекпоинты моделей (`.ckpt`)
- `outputs/` — метрики (`metrics.json`), переводы (`predictions.txt`)

Remote уже настроен в .dvc/config. Всё, что требуется — положить в корень проекта файл **`dvc-sa.json`**.

---

## Использование

В /notebooks лежит .ipynb ноутбок в котором находится пример использования
