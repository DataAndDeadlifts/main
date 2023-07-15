import click
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")

tqdm.pandas()


from llm_mito_scanner.training.transcription.generation import \
    get_training_sequences_with_idx


def update_pbar(result, pbar):
    pbar.update(1)
    return result


def get_chromosome_sequences_with_idx(args: dict) -> tuple[int, pd.DataFrame]:
    process_idx = next(iter(mp.current_process()._identity), 1)
    index = args.get('index') # this is the index for a single chromosome
    chromosome = args.get("chromosome")
    genes_path = args.get("genes_path")
    intron_location_path = args.get('intron_locations_path')
    pbar = tqdm(
        total=index.shape[0], position=process_idx, ncols=80, 
        desc=f"{chromosome}", leave=False
        )
    index_geneids = index.geneid.unique()
    index_transcriptids = index.transcriptid.unique()
    intron_locations = pd.read_parquet(intron_location_path / f"chromosome-{chromosome}.parquet")
    intron_locations.loc[:, 'chromosome'] = chromosome
    intron_locations = intron_locations[
        (intron_locations.geneid.isin(index_geneids)) &
        (intron_locations.transcriptid.isin(index_transcriptids))
    ]
    chromosome_genes = pd.read_csv(genes_path / f"{chromosome}.csv")
    chromosome_genes = chromosome_genes[
        (chromosome_genes.geneid.isin(index_geneids))
    ]
    training_sequences = index.apply(
        lambda row: update_pbar(
            get_training_sequences_with_idx(
                row.chromosome, row.geneid, row.transcriptid, 
                chromosome_genes,
                intron_locations,
                row.start, row.end
            ), 
            pbar), 
        axis=1).values.tolist()
    training_sequence_frame = pd.DataFrame(training_sequences, columns=['input', 'target', 'position'])
    training_sequence_frame.loc[:, 'chromosome'] = chromosome
    training_sequence_frame.loc[:, 'type'] = index['type']
    pbar.close()
    return chromosome, training_sequence_frame


@click.command()
@click.argument("assembly-path", type=Path)
def generate_sequences(assembly_path: Path):
    genes_path = assembly_path / "genes"
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    intron_locations_path = transcription_data_path / "intron_positions"
    index_path = transcription_data_path / "training_sequence_idx.csv"
    index = pd.read_csv(index_path)
    tasks = [{
        "index": index[index.chromosome == chromosome],
        "chromosome": chromosome,
        "genes_path": genes_path,
        "intron_locations_path": intron_locations_path} for chromosome in index.chromosome.unique()
    ]
    partition_path = transcription_data_path / "sequences"
    if not partition_path.exists():
        partition_path.mkdir()
    pool = mp.Pool(processes=6)
    try:
        pbar = tqdm(total=len(tasks), ncols=80, desc="Generating")
        for chromosome, frame in pool.imap_unordered(get_chromosome_sequences_with_idx, tasks):
            epoch_path = partition_path / f"{chromosome}.parquet"
            frame.to_parquet(epoch_path)
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    generate_sequences()
