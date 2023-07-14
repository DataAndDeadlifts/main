import click
from pathlib import Path
from tqdm import tqdm
from torch.optim import Adam
import torch
import torch.nn as nn
import warnings

warnings.simplefilter("ignore")

from llm_mito_scanner.training.transcription.generation import PAD_TOK
from llm_mito_scanner.training.transcription.train import get_vocab, \
    train_epoch, evaluate, Seq2SeqTransformer, DEVICE, set_text_transform, \
        set_vocab_idx, TranscriptionDataset


@click.command()
@click.argument("assembly_path", type=Path)
@click.option("--epochs", type=int, default=1)
@click.option("--embedding_size", type=int, default=32)
@click.option("--nheads", type=int, default=4)
@click.option("--feed-forward-dim", type=int, default=32)
@click.option("--encoder-layers", type=int, default=1)
@click.option("--decoder-layers", type=int, default=1)
@click.option("--batch-size", type=int, default=32)
@click.option("--batch-limit", type=int, default=None)
def train(assembly_path: Path, 
          epochs: int, 
          embedding_size: int, nheads: int, feed_forward_dim: int,
          encoder_layers: int, decoder_layers: int,
          batch_size: int, batch_limit: int | None):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    # Load vocab
    vocab = get_vocab(transcription_data_path)
    # Set properties of vocab
    set_text_transform(vocab)
    set_vocab_idx(vocab)
    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = len(vocab)
    # Define model
    transformer = Seq2SeqTransformer(
        encoder_layers, decoder_layers, embedding_size,
        nheads, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, feed_forward_dim)
    # Initialize weights, send to GPU
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)
    # Define loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOK])
    # Define optimizer
    optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    # Start training
    training_pbar = tqdm(total=epochs, leave=True, ncols=80)
    # Save state every epoch
    checkpoint_path = transcription_data_path / "checkpoints"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    state = {
        'epoch': 0,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': None,
        'eval': None
    }
    for epoch in range(1, epochs+1):
        state['epoch'] = epoch
        state['loss'] = None
        state['eval'] = None
        try:
            training_pbar.set_description("Training")
            train_dataset = TranscriptionDataset(transcription_data_path / "training_data.csv", True)
            train_loss = train_epoch(
                transformer, optimizer, loss_fn, 
                train_dataset,
                batch_size=batch_size,
                limit=batch_limit)
            state['loss'] = train_loss
        except KeyboardInterrupt:
            pass
        try:
            training_pbar.set_description("Evaluating")
            eval_dataset = TranscriptionDataset(transcription_data_path / "training_data.csv", False)
            val_loss = evaluate(
                transformer, loss_fn,
                eval_dataset,
                batch_size=batch_size,
                limit=batch_limit)
            state['eval'] = val_loss
        except KeyboardInterrupt:
            pass
        epoch_checkpoint_path = checkpoint_path / f"epoch-{epoch}-model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'eval': val_loss
            }, epoch_checkpoint_path)
        training_pbar.set_description(f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        training_pbar.update(1)


if __name__ == "__main__":
    train()
