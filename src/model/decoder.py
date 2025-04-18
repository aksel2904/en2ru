import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    GRU Decoder с вниманием (additive attention).

    Args:
        output_dim (int): размер словаря выходного языка
        emb_dim (int): размерность эмбеддинга
        enc_hid_dim (int): скрытое состояние энкодера (в одну сторону)
        dec_hid_dim (int): скрытое состояние декодера
        attention (nn.Module): модуль внимания (Attention)
    """

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(input_size=emb_dim + enc_hid_dim * 2, hidden_size=dec_hid_dim)

        self.out = nn.Linear(
            in_features=attention.attn_in + emb_dim, out_features=output_dim
        )

    def weighted_encoder(self, decoder_hidden, encoder_outputs):
        """
        Вычисляет взвешенное представление encoder_outputs на основе attention-скор.

        Args:
            decoder_hidden (Tensor): [batch, dec_hid_dim]
            encoder_outputs (Tensor): [src_len, batch, enc_hid_dim * 2]

        Returns:
            context (Tensor): [1, batch, enc_hid_dim * 2]
        """
        a = self.attention(decoder_hidden, encoder_outputs)  # [batch, src_len]
        a = a.unsqueeze(1)  # [batch, 1, src_len]

        encoder_outputs = encoder_outputs.permute(
            1, 0, 2
        )  # [batch, src_len, enc_hid_dim * 2]

        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid_dim * 2]
        return weighted.permute(1, 0, 2)  # [1, batch, enc_hid_dim * 2]

    def forward(self, input, decoder_hidden, encoder_outputs):
        """
        Args:
            input (Tensor): [batch] — токен предыдущего времени
            decoder_hidden (Tensor): [batch, dec_hid_dim]
            encoder_outputs (Tensor): [src_len, batch, enc_hid_dim * 2]

        Returns:
            output_logits (Tensor): [batch, output_dim]
            new_hidden (Tensor): [batch, dec_hid_dim]
        """
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.embedding(input)  # [1, batch, emb_dim]

        weighted = self.weighted_encoder(
            decoder_hidden, encoder_outputs
        )  # [1, batch, enc_hid_dim * 2]
        rnn_input = torch.cat(
            (embedded, weighted), dim=2
        )  # [1, batch, emb_dim + enc_hid_dim * 2]

        output, hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0)
        )  # output: [1, batch, dec_hid_dim]

        output = output.squeeze(0)  # [batch, dec_hid_dim]
        weighted = weighted.squeeze(0)  # [batch, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)  # [batch, emb_dim]

        prediction = self.out(
            torch.cat((output, weighted, embedded), dim=1)
        )  # [batch, output_dim]

        return prediction, hidden.squeeze(0)
