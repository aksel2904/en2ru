import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence


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


    def inference_beam_search(self, src, bos_id, eos_id, pad_id, beam_width=5, max_len=None):
        """
        Beam search decoding.

        Args:
            src (Tensor): [src_len, batch]
            bos_id (int)
            eos_id (int)
            pad_id (int)
            beam_width (int)
            max_len (int, optional)

        Returns:
            Tensor: [batch, seq_len]
        """
        batch_size = src.shape[1]
        max_len = max_len or (src.shape[0] * 2)

        encoder_outputs, hidden = self.encoder(src)

        beams = [[{
            'tokens': [bos_id],
            'score': 0.0,
            'hidden': hidden[:, i:i+1].clone(),
            'finished': False
        }] for i in range(batch_size)]

        for step in range(max_len):
            all_candidates = [[] for _ in range(batch_size)]

            for batch_idx in range(batch_size):
                current_beams = beams[batch_idx]

                for beam in current_beams:
                    if beam['finished']:
                        all_candidates[batch_idx].append(beam)
                        continue

                    last_token = torch.tensor([beam['tokens'][-1]], device=self.device)
                    current_hidden = beam['hidden']

                    output, new_hidden = self.decoder(
                        last_token,
                        current_hidden.squeeze(0),
                        encoder_outputs[:, batch_idx:batch_idx+1, :]
                    )

                    log_probs = F.log_softmax(output, dim=1)
                    topk_scores, topk_ids = log_probs.topk(beam_width, dim=1)

                    for score, token_id in zip(topk_scores[0], topk_ids[0]):
                        new_beam = {
                            'tokens': beam['tokens'] + [token_id.item()],
                            'score': beam['score'] + score.item(),
                            'hidden': new_hidden.clone(),
                            'finished': (token_id.item() == eos_id)
                        }
                        all_candidates[batch_idx].append(new_beam)

            new_beams = []
            for batch_idx in range(batch_size):
                candidates = all_candidates[batch_idx]
                candidates.sort(key=lambda x: x['score'] / (len(x['tokens'])**0.7), reverse=True)
                new_beams.append(candidates[:beam_width])

            beams = new_beams

            if all(all(b['finished'] for b in beam_group) for beam_group in beams):
                break

        final_outputs = []
        for beam_group in beams:
            if not beam_group:
                final_outputs.append(torch.tensor([eos_id], device=self.device))
                continue
            best = max(beam_group, key=lambda x: x['score'] / (len(x['tokens'])**0.7))
            final_outputs.append(torch.tensor(best['tokens'][1:], device=self.device))  # remove <bos>

        return pad_sequence(final_outputs, batch_first=True, padding_value=pad_id)