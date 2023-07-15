# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/04 training.transcription.train.ipynb.

# %% auto 0
__all__ = ['DEVICE', 'UNK_IDX', 'PAD_IDX', 'BOS_IDX', 'EOS_IDX', 'text_transform', 'list_string_to_list',
           'count_tokens_in_sequence_file', 'build_vocab', 'get_vocab', 'set_vocab_idx', 'PositionalEncoding',
           'TokenEmbedding', 'Seq2SeqTransformer', 'sequential_transforms', 'tensor_transform', 'set_text_transform',
           'collate_fn', 'generate_square_subsequent_mask', 'create_mask', 'train_epoch', 'evaluate', 'greedy_decode',
           'translate']

# %% ../../../nbs/04 training.transcription.train.ipynb 1
import os
from pathlib import Path
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import vocab, Vocab
from sklearn.model_selection import train_test_split
import math
from timeit import default_timer as timer
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict, Counter
from multiprocessing import Pool

from llm_mito_scanner.data.download import load_config, \
    get_latest_assembly_path
from llm_mito_scanner.training.transcription.generation import \
    BOS_TOK, EOS_TOK, UNK_TOK, PAD_TOK


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% ../../../nbs/04 training.transcription.train.ipynb 6
def list_string_to_list(l: str, sep=", ") -> list[str]:
    return l.replace("'", "").strip("][").split(sep)

# %% ../../../nbs/04 training.transcription.train.ipynb 7
def count_tokens_in_sequence_file(path: Path) -> Counter:
    token_counter = Counter()
    for col in ['input', 'target']:
        sequences = pd.read_parquet(path, columns=[col])[col].tolist()
        sequences_counters = list(map(Counter, sequences))
        token_counter = token_counter + sum(sequences_counters, Counter())
    return token_counter


def build_vocab(transcription_data_path: Path, special_tokens: list[str] = [UNK_TOK, BOS_TOK, EOS_TOK, PAD_TOK]) -> Vocab:
    sequences_path = transcription_data_path / "sequences"
    parquet_files = list(sequences_path.glob("*.parquet"))
    pbar = tqdm(total=len(parquet_files), ncols=80, leave=False, desc="Building Vocab")
    pool = Pool(os.cpu_count() - 1)
    token_counter = Counter()
    try:
        for c in pool.imap_unordered(count_tokens_in_sequence_file, parquet_files):
            token_counter = token_counter + c
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()
    token_vocab = vocab(OrderedDict(token_counter.most_common()), specials=special_tokens)
    token_vocab.set_default_index(token_vocab[UNK_TOK])
    return token_vocab


def get_vocab(transcription_data_path: Path, special_tokens: list[str] = [UNK_TOK, BOS_TOK, EOS_TOK, PAD_TOK]) -> Vocab:
    vocab_path = transcription_data_path / "vocab.pt"
    if vocab_path.exists():
        token_vocab = torch.load(vocab_path)
        return token_vocab
    else:
        token_vocab = build_vocab(transcription_data_path, special_tokens)
        torch.save(token_vocab, vocab_path)
        return token_vocab

# %% ../../../nbs/04 training.transcription.train.ipynb 11
UNK_IDX = None
PAD_IDX = None
BOS_IDX = None
EOS_IDX = None

def set_vocab_idx(vocab: Vocab):
    global UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX
    UNK_IDX = vocab[UNK_TOK]
    PAD_IDX = vocab[PAD_TOK]
    BOS_IDX = vocab[BOS_TOK]
    EOS_IDX = vocab[EOS_TOK]

# %% ../../../nbs/04 training.transcription.train.ipynb 14
class TranscriptionDataset(Dataset):
    def __init__(self, path: Path, train: bool, **train_test_split_kwargs) -> None:
        self.path = path
        self.train = train
        data_train, data_test = train_test_split(pd.read_parquet(path), **train_test_split_kwargs)
        self.data = data_train if train else data_test
    
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> tuple[str, str, int]:
        row = self.data.iloc[idx]
        input_str = list_string_to_list(row.input)
        target_str = list_string_to_list(row.target)
        position = row.position
        return input_str, target_str, position
    
    def save_index(self, path: Path):
        

# %% ../../../nbs/04 training.transcription.train.ipynb 17
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

# %% ../../../nbs/04 training.transcription.train.ipynb 19
# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# %% ../../../nbs/04 training.transcription.train.ipynb 20
text_transform = None
def set_text_transform(
        text_vocab: Vocab
):
    global text_transform
    text_transform = sequential_transforms(
    text_vocab, # Numericalization
    tensor_transform)

# %% ../../../nbs/04 training.transcription.train.ipynb 23
# function to collate data samples into batch tensors
def collate_fn(batch: list[tuple[str, str, int]]):
    global PAD_IDX
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample, pos in batch:
        src_batch.append(text_transform(src_sample))
        tgt_batch.append(text_transform(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# %% ../../../nbs/04 training.transcription.train.ipynb 35
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# %% ../../../nbs/04 training.transcription.train.ipynb 44
def train_epoch(
        model, optimizer, loss_fn, 
        dataset: TranscriptionDataset,
        batch_size: int = 32,
        pbar_position: int = 1,
        limit: int = None):
    global DEVICE
    model.train()
    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    batches = len(train_dataloader)
    batch_pbar = tqdm(total=batches, position=pbar_position, leave=False, ncols=80, desc="Training")
    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        batch_pbar.update(1)
        if isinstance(limit, int):
            if i >= limit:
                batches = limit
                break

    batch_pbar.close()

    return losses / batches


def evaluate(
        model, loss_fn, 
        dataset: TranscriptionDataset,
        batch_size: int = 32,
        pbar_position: int = 1,
        limit: int = None):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    batches = len(val_dataloader)
    eval_pbar = tqdm(total=batches, position=pbar_position, leave=False, ncols=80, desc="Evaluating")
    for i, (src, tgt) in enumerate(val_dataloader):
        # Break these up into reasonable sizes for the GPU
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        
        eval_pbar.update(1)
        if isinstance(limit, int):
            if i >= limit:
                batches = limit
                break
        
    eval_pbar.close()
    return losses / batches

# %% ../../../nbs/04 training.transcription.train.ipynb 46
# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, vocab: Vocab, src_sentence: str):
    model.eval()
    src = text_transform(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
