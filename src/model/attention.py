import torch
import torch.nn as nn
import torch.nn.functional as F


'''class Attention(nn.Module):
    """
    Attention: tanh → sum по последнему измерению → softmax
    """

    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden (Tensor): [batch_size, dec_hid_dim]
            encoder_outputs (Tensor): [src_len, batch_size, enc_hid_dim * 2]

        Returns:
            attention_weights (Tensor): [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[0]

        # [batch, src_len, dec_hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # [batch, src_len, enc_hid_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # [batch, src_len, enc_hid_dim * 2 + dec_hid_dim]
        concat = torch.cat((decoder_hidden, encoder_outputs), dim=2)

        # [batch, src_len, attn_dim]
        energy = torch.tanh(self.attn(concat))

        # [batch, src_len]
        attention = energy.sum(dim=2)

        return F.softmax(attention, dim=1)
'''
class Attention(nn.Module):
    """
    Additive (Bahdanau-style) Attention mechanism.

    Args:
        enc_hid_dim (int): Hidden size of encoder (per direction)
        dec_hid_dim (int): Hidden size of decoder
        attn_dim (int): Dimension of attention intermediate layer
    """
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden (Tensor): [batch_size, dec_hid_dim]
            encoder_outputs (Tensor): [src_len, batch_size, enc_hid_dim * 2]

        Returns:
            attention_weights (Tensor): [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[0]

        # [batch_size, src_len, dec_hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # [batch_size, src_len, enc_hid_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # concat decoder hidden and encoder outputs
        concat = torch.cat((decoder_hidden, encoder_outputs), dim=2)  # [batch_size, src_len, attn_in]
        energy = torch.tanh(self.attn(concat))                        # [batch_size, src_len, attn_dim]
        attention = self.v(energy).squeeze(2)                        # [batch_size, src_len]

        return F.softmax(attention, dim=1)                           # normalized weights
