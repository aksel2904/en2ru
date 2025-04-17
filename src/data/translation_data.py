import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm


class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, sp_src, sp_tgt):
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f]

        self.sp_src = sp_src
        self.sp_tgt = sp_tgt

    def __getitem__(self, idx):
        src = self.sp_src.encode(self.src_sentences[idx], out_type=int)
        tgt = self.sp_tgt.encode(self.tgt_sentences[idx], out_type=int)

        # Добавляем <bos> и <eos>
        src = [self.sp_src.bos_id()] + src + [self.sp_src.eos_id()]
        tgt = [self.sp_tgt.bos_id()] + tgt + [self.sp_tgt.eos_id()]

        return torch.tensor(src), torch.tensor(tgt)

    def __len__(self):
        return len(self.src_sentences)


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_src,
        train_tgt,
        val_src,
        val_tgt,
        test_src,
        test_tgt,
        sp_model_src,
        sp_model_tgt,
        batch_size=32,
        num_workers=0 #?
    ):
        super().__init__()
        self.train_src = train_src
        self.train_tgt = train_tgt
        self.val_src = val_src
        self.val_tgt = val_tgt
        self.test_src = test_src
        self.test_tgt = test_tgt
        self.sp_model_src = sp_model_src
        self.sp_model_tgt = sp_model_tgt
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.sp_src = spm.SentencePieceProcessor()
        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_src.load(self.sp_model_src)
        self.sp_tgt.load(self.sp_model_tgt)

        self.pad_id_src = self.sp_src.pad_id()
        self.pad_id_tgt = self.sp_tgt.pad_id()

        self.train_data = TranslationDataset(
            self.train_src, self.train_tgt, self.sp_src, self.sp_tgt
        )
        self.val_data = TranslationDataset(
            self.val_src, self.val_tgt, self.sp_src, self.sp_tgt
        )
        self.test_data = TranslationDataset(
            self.test_src, self.test_tgt, self.sp_src, self.sp_tgt
        )

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=self.pad_id_src)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_id_tgt)
        '''
        batch_first=True?
        '''
        return src_batch, tgt_batch

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    @property
    def input_vocab_size(self):
        return self.sp_src.get_piece_size()

    @property
    def target_vocab_size(self):
        return self.sp_tgt.get_piece_size()
