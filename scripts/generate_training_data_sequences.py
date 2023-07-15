import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
from multiprocessing import Pool

warnings.simplefilter("ignore")

tqdm.pandas()

from llm_mito_scanner.training.transcription.sampling import get_chromosome_idx_sequences, get_mrna_file_index


@click.command()
@click.argument("assembly-path", type=Path)
@click.option("--chunk-size", type=int, default=200)
@click.option("--write-size", type=int, default=500000)
def generate_sequences(assembly_path: Path, chunk_size: int, write_size: int):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    index_path = transcription_data_path / "training_sequence_idx.csv"
    index = pd.read_csv(index_path)
    mrna_file_index = get_mrna_file_index(transcription_data_path)
    index.loc[:, 'mrna_partition'] = index.apply(
        lambda row: mrna_file_index.get((row.chromosome, row.geneid, row.transcriptid)),
        axis=1)
    del mrna_file_index
    tasks = [
        {
            "chromosome": chromosome,
            "index": index[index.chromosome == chromosome],
            "assembly": assembly_path,
            "position": i + 1,
            "chunk_size": chunk_size,
            "write_size": write_size,
            "save": True
        } for i, chromosome in enumerate(index.chromosome.unique().tolist())
    ]
    del index
    partition_path = transcription_data_path / "sequences"
    if not partition_path.exists():
        partition_path.mkdir()
    pool = Pool(1)
    task_pbar = tqdm(tasks, ncols=80, desc="Generating", leave=False)
    try:
        for _ in pool.imap_unordered(get_chromosome_idx_sequences, tasks):
            task_pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        task_pbar.close()
    


if __name__ == "__main__":
    generate_sequences()
