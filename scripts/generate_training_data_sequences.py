import click
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")

tqdm.pandas()

from llm_mito_scanner.training.transcription.sampling import get_chromosome_idx_sequences, get_mrna_file_index

def update_pbar(result, pbar):
    pbar.update(1)
    return result


@click.command()
@click.argument("assembly-path", type=Path)
def generate_sequences(assembly_path: Path):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    index_path = transcription_data_path / "training_sequence_idx.csv"
    index = pd.read_csv(index_path)
    mrna_file_index = get_mrna_file_index(transcription_data_path)
    index.loc[:, 'mrna_partition'] = index.apply(
        lambda row: mrna_file_index.get((row.chromosome, row.geneid, row.transcriptid)),
        axis=1)
    tasks = [
        {
            "chromosome": chromosome,
            "index": index[index.chromosome == chromosome],
            "assembly": assembly_path,
            "mrna_file_index": mrna_file_index,
            "position": i + 1,
            "chunk_size": 500,
            "save": True
        } for i, chromosome in enumerate(index.chromosome.unique().tolist())
    ]
    partition_path = transcription_data_path / "sequences"
    if not partition_path.exists():
        partition_path.mkdir()
    pool = mp.Pool(processes=4)
    try:
        pbar = tqdm(total=len(tasks), ncols=80, desc="Generating")
        for _ in pool.imap_unordered(get_chromosome_idx_sequences, tasks):
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    generate_sequences()
