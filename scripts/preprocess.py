import hydra
from omegaconf import DictConfig
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sentencepiece as spm


@hydra.main(config_path="../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    en_path = cfg.data.raw_en
    ru_path = cfg.data.raw_ru
    out_dir = cfg.data.out_dir

    # 1. Сплит
    with open(en_path, "r") as f:
        en = [line.strip().lower() for line in f]
    with open(ru_path, "r") as f:
        ru = [line.strip().lower() for line in f]
    df = pd.DataFrame({"en": en, "ru": ru})
    train_val, test = train_test_split(
        df, test_size=cfg.split.test_size, random_state=cfg.split.seed
    )
    train, val = train_test_split(
        train_val,
        test_size=cfg.split.val_size / (1 - cfg.split.test_size),
        random_state=cfg.split.seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    for name, data in zip(["train", "val", "test"], [train, val, test]):
        data["en"].to_csv(f"{out_dir}/{name}.en", index=False, header=False)
        data["ru"].to_csv(f"{out_dir}/{name}.ru", index=False, header=False)

    # 2. BPE
    spm.SentencePieceTrainer.train(
        input=f"{out_dir}/train.en",
        model_prefix=f"{out_dir}/bpe_en",
        vocab_size=cfg.bpe.vocab_size_en,
        user_defined_symbols=["<bos>", "<eos>"],
        pad_id=1,
        unk_id=0,
        bos_id=2,
        eos_id=3,
        model_type="bpe",
        character_coverage=1.0,
    )
    spm.SentencePieceTrainer.train(
        input=f"{out_dir}/train.ru",
        model_prefix=f"{out_dir}/bpe_ru",
        vocab_size=cfg.bpe.vocab_size_ru,
        user_defined_symbols=["<bos>", "<eos>"],
        pad_id=1,
        unk_id=0,
        bos_id=2,
        eos_id=3,
        model_type="bpe",
        character_coverage=1.0,
    )


if __name__ == "__main__":
    main()
