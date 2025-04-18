import os
import torch
import re
import hydra
import sacrebleu
from omegaconf import DictConfig
from data.translation_data import TranslationDataModule
from model.encoder import Encoder
from model.attention import Attention
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from model.lightning_module import Seq2SeqLightningModule


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for src_batch, _ in data_loader:
            src_batch = src_batch.to(device)
            predicted_tokens = model.model.inference(
                src_batch, bos_id=model.pad_id + 1, eos_id=model.pad_id + 2
            )
            # predicted_tokens = model.model.inference_beam_search(src_batch, bos_id=model.pad_id + 1, eos_id=model.pad_id + 2, pad_id=model.pad_id)
            for seq in predicted_tokens:
                predictions.append(seq)

    return predictions


def indices_to_text(preds, sp_model):
    texts = []

    for indices in preds:
        indices_list = indices.tolist() if not isinstance(indices, list) else indices

        # убираем дубли
        filtered = []
        prev = None
        for idx in indices_list:
            if idx != prev:
                filtered.append(idx)
                prev = idx

        text = sp_model.decode_ids(filtered)
        text = (
            text.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").strip()
        )
        text = re.sub(r"([^\w\s])\1+", r"\1", text)
        texts.append(text)

    return texts


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data = TranslationDataModule(
        train_src=cfg.data.train_src,
        train_tgt=cfg.data.train_tgt,
        val_src=cfg.data.val_src,
        val_tgt=cfg.data.val_tgt,
        test_src=cfg.data.test_src,
        test_tgt=cfg.data.test_tgt,
        sp_model_src=cfg.data.sp_model_src,
        sp_model_tgt=cfg.data.sp_model_tgt,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    data.setup()

    input_dim = data.input_vocab_size
    output_dim = data.target_vocab_size

    # Model
    enc = Encoder(
        input_dim, cfg.model.enc_emb_dim, cfg.model.enc_hid_dim, cfg.model.dec_hid_dim
    )
    attn = Attention(cfg.model.enc_hid_dim, cfg.model.dec_hid_dim, cfg.model.attn_dim)
    dec = Decoder(
        output_dim,
        cfg.model.dec_emb_dim,
        cfg.model.enc_hid_dim,
        cfg.model.dec_hid_dim,
        attn,
    )

    model = Seq2SeqLightningModule(
        enc, dec, sp_tgt=data.sp_tgt, learning_rate=cfg.model.learning_rate
    )
    model.load_state_dict(torch.load("weights/best.ckpt")["state_dict"])
    model.to(device)

    # Get predictions
    predictions = evaluate(model, data.test_dataloader(), device)
    decoded = indices_to_text(predictions, data.sp_tgt)

    print("Примеры перевода:")
    for i in range(min(5, len(decoded))):
        print(f"{i + 1}. {decoded[i]}")

    # можно сохранить в файл:
    with open("outputs/predictions.txt", "w") as f:
        f.writelines(line + "\n" for line in decoded)

    # Загружаем референсы
    ref_path = cfg.data.test_tgt
    with open(ref_path, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f.readlines()]

    # BLEU вычисляется по токенизированным предложениям
    bleu = sacrebleu.corpus_bleu(decoded, [references], tokenize="none")
    print(f"\nBLEU score: {bleu.score:.2f}")


if __name__ == "__main__":
    main()
