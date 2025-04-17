import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder with embedding and linear projection to decoder hidden state.
    
    Args:
        input_dim (int): vocabulary size (source language)
        emb_dim (int): embedding size
        enc_hid_dim (int): hidden size of GRU (each direction)
        dec_hid_dim (int): hidden size of decoder (output projection)
    """
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            bidirectional=True
        )

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        """
        Args:
            src (Tensor): [src_len, batch_size]
        
        Returns:
            outputs (Tensor): [src_len, batch_size, enc_hid_dim * 2]
            hidden (Tensor): [batch_size, dec_hid_dim]
        """
        embedded = self.embedding(src)  # [src_len, batch_size, emb_dim]

        outputs, hidden = self.rnn(embedded)
        # hidden: [2, batch_size, enc_hid_dim] → bidirectional → 2 layers

        # concat last forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, enc_hid_dim * 2]

        hidden_proj = torch.tanh(self.fc(hidden_cat))  # [batch_size, dec_hid_dim]

        return outputs, hidden_proj
