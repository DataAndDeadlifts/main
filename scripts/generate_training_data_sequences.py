import click
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from multiprocessing import Pool

warnings.simplefilter("ignore")

tqdm.pandas()

from llm_mito_scanner.training.transcription.sampling import get_multiple_training_sequences_wrapper


@click.command()
@click.argument("assembly-path", type=Path)
@click.option("--epochs", type=int, default=10)
@click.option("--chunk-size", type=int, default=50)
@click.option("--batch-size", type=int, default=10000)
def generate_sequences(assembly_path: Path, epochs: int, chunk_size: int, batch_size: int):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    index_path = transcription_data_path / "training_sequence_idx.csv"
    index = pd.read_csv(index_path)
    index.loc[:, "idx"] = pd.Series(zip(index.start, index.end))
    index = index.groupby(["chromosome", "geneid", "transcriptid"]).agg({
        "idx": list,
        "type": list
    })
    index.loc[:, 'samples'] = index.idx.apply(len)
    index.sort_values('samples', ascending=False, inplace=True)
    index.reset_index(drop=False, inplace=True)
    index = np.array_split(index, epochs)
    tasks = [
        {
            "index": i,
            "assembly_path": assembly_path,
            "epoch": e + 1,
            "chunk_size": chunk_size,
            "batch_size": batch_size
        } for e, i in enumerate(index)
    ]
    del index
    partition_path = transcription_data_path / "sequences"
    if not partition_path.exists():
        partition_path.mkdir()
    pool = Pool(4)
    task_pbar = tqdm(tasks, ncols=80, desc="Generating", leave=False)
    try:
        for _ in pool.imap_unordered(get_multiple_training_sequences_wrapper, tasks):
            # Write frame
            task_pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        task_pbar.close()
    

if __name__ == "__main__":
    generate_sequences()
