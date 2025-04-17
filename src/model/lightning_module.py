import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.seq2seq import Seq2Seq


class Seq2SeqLightningModule(pl.LightningModule):
    def __init__(self, encoder, decoder, sp_tgt, learning_rate=1e-3):
        super().__init__()

        self.model = Seq2Seq(encoder, decoder, device=self.device)
        self.learning_rate = learning_rate
        self.pad_id = sp_tgt.pad_id()

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        return self.model(src, trg, teacher_forcing_ratio)

    def training_step(self, batch, batch_idx):
        src, trg = batch  # [src_len, batch], [trg_len, batch]
        output = self(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)

        loss = self.criterion(output, trg)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output = self(src, trg, teacher_forcing_ratio=0.0)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)

        loss = self.criterion(output, trg)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
