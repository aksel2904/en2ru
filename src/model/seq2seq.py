import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    """
    Seq2Seq модель с вниманием и поддержкой teacher forcing и inference.

    Args:
        encoder (Encoder): модуль энкодера
        decoder (Decoder): модуль декодера
        device (torch.device): устройство (cuda или cpu)
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = "cuda" # temporary. was: = device

    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        """
        Обучающий проход с teacher forcing.

        Args:
            src (Tensor): [src_len, batch]
            trg (Tensor): [trg_len, batch]
            teacher_forcing_ratio (float): вероятность использовать ground truth токен

        Returns:
            outputs (Tensor): [trg_len, batch, vocab_size]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        input_token = trg[0, :]  # <bos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[t] = output

            top1 = output.argmax(1)  # [batch]

            teacher_force = random.random() < teacher_forcing_ratio
            input_token = trg[t] if teacher_force else top1

        return outputs

    def inference(self, src, bos_id, eos_id, max_len=None):
        """
        Inference с жадным поиском (greedy decoding).

        Args:
            src (Tensor): [src_len, batch]
            bos_id (int): индекс <bos> токена
            eos_id (int): индекс <eos> токена
            max_len (int, optional): максимальная длина (по умолчанию 2 * src_len)

        Returns:
            predictions (Tensor): [batch, seq_len] — сгенерированные индексы
        """
        batch_size = src.shape[1]
        max_len = max_len or (src.shape[0] * 2)

        encoder_outputs, hidden = self.encoder(src)
        input_token = torch.tensor([bos_id] * batch_size).to(self.device)

        predictions = []

        for _ in range(max_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)  # [batch]
            predictions.append(top1)

            input_token = top1

            if (top1 == eos_id).all():
                break

        predictions = torch.stack(predictions, dim=1)  # [batch, seq_len]
        return predictions
