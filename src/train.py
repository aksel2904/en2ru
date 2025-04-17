import torch
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl

from data.translation_data import TranslationDataModule
from model.encoder import Encoder
from model.attention import Attention
from model.decoder import Decoder
from model.lightning_module import Seq2SeqLightningModule
import os


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    print("🚀 Запуск обучения с конфигом:")
    print(OmegaConf.to_yaml(cfg))

    # DataModule
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
        num_workers=cfg.data.num_workers
    )
    data.setup()

    # vocab sizes
    input_dim = data.input_vocab_size
    output_dim = data.target_vocab_size

    # Model init
    enc = Encoder(
        input_dim=input_dim,
        emb_dim=cfg.model.enc_emb_dim,
        enc_hid_dim=cfg.model.enc_hid_dim,
        dec_hid_dim=cfg.model.dec_hid_dim
    )

    attn = Attention(
        enc_hid_dim=cfg.model.enc_hid_dim,
        dec_hid_dim=cfg.model.dec_hid_dim,
        attn_dim=cfg.model.attn_dim
    )

    dec = Decoder(
        output_dim=output_dim,
        emb_dim=cfg.model.dec_emb_dim,
        enc_hid_dim=cfg.model.enc_hid_dim,
        dec_hid_dim=cfg.model.dec_hid_dim,
        attention=attn
    )

    model = Seq2SeqLightningModule(
        encoder=enc,
        decoder=dec,
        sp_tgt=data.sp_tgt,
        learning_rate=cfg.model.learning_rate
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir="weights/",  # можно будет пушить через DVC
        log_every_n_steps=20
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
