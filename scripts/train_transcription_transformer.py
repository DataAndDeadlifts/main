import click
from pathlib import Path
from tqdm import tqdm
from torch.optim import Adam
import torch
import torch.nn as nn
import warnings

from llm_mito_scanner.training.transcription.index import get_vocab, PAD_TOKEN
from llm_mito_scanner.training.transcription.train import train_epoch, \
    evaluate, Seq2SeqTransformer, DEVICE, set_text_transform, set_vocab_idx


warnings.simplefilter("ignore")


@click.command()
@click.argument("assembly_path", type=Path)
@click.option("--epochs", type=int, default=1)
@click.option("--embedding_size", type=int, default=16)
@click.option("--nheads", type=int, default=4)
@click.option("--feed-forward-dim", type=int, default=16)
@click.option("--encoder-layers", type=int, default=1)
@click.option("--decoder-layers", type=int, default=1)
@click.option("--batch-size", type=int, default=32)
@click.option("--collate-size", type=int, default=1500)
@click.option("--chromosome", type=str, default=None)
def train(assembly_path: Path, 
          epochs: int, 
          embedding_size: int, nheads: int, feed_forward_dim: int,
          encoder_layers: int, decoder_layers: int,
          batch_size: int, collate_size: int,
          chromosome: str | None):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    sequences_data_path = transcription_data_path / "sequences"
    # Load vocab
    vocab = get_vocab(transcription_data_path)
    set_text_transform(vocab)
    set_vocab_idx(vocab)

    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = len(vocab)

    PAD_IDX = vocab[PAD_TOKEN]

    transformer = Seq2SeqTransformer(
        encoder_layers, decoder_layers, embedding_size,
        nheads, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, feed_forward_dim)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    training_pbar = tqdm(total=epochs, leave=True, ncols=80)
    for epoch in range(1, epochs+1):
        train_loss = None
        try:
            training_pbar.set_description("Training")
            train_loss = train_epoch(
                transformer, optimizer, loss_fn, 
                transcription_data_path, sequences_data_path,
                batch_size=batch_size,
                collate_size=collate_size,
                chromosome=chromosome)
        except KeyboardInterrupt:
            pass
        try:
            training_pbar.set_description("Evaluating")
            val_loss = evaluate(
                transformer, loss_fn,
                transcription_data_path, sequences_data_path,
                batch_size=batch_size,
                collate_size=collate_size,
                chromosome=chromosome)
        except KeyboardInterrupt:
            pass
        if train_loss is not None:
            checkpoint_path = training_data_path / f"epoch_{epoch}-model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, checkpoint_path)
        training_pbar.set_description(f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        training_pbar.update(1)


if __name__ == "__main__":
    train()
