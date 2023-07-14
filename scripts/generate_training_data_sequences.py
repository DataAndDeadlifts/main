import click
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

tqdm.pandas()


from llm_mito_scanner.data.transcription import read_all_chromosome_gene_info
from llm_mito_scanner.training.transcription.index import get_intron_locations
from llm_mito_scanner.training.transcription.generation import \
    get_training_sequences_with_idx


def update_pbar(result, pbar):
    pbar.update(1)
    return result


def get_chromosome_sequences_with_idx(args: dict) -> pd.DataFrame:
    process_idx = next(iter(mp.current_process()._identity), 0)
    index = args.get('index') # this is the index for a single chromosome
    chromosome = args.get("chromosome")
    genes_path = args.get("genes_path")
    chromosome_intron_location_path = args.get('intron_locations_path')
    chromosome_intron_locations = pd.read_parquet(chromosome_intron_location_path)
    chromosome_intron_locations.loc[:, 'chromosome'] = chromosome_intron_location_path.stem.replace("chromosome-", "")
    chromosome_genes = pd.read_csv(genes_path / f"{chromosome}.csv")
    pbar = tqdm(total=index.shape[0], position=process_idx, ncols=80, desc=f"{chromosome}", leave=False)
    training_sequences = index.apply(
        lambda row: update_pbar(
            get_training_sequences_with_idx(
                row.chromosome, row.geneid, row.transcriptid, 
                chromosome_genes,
                chromosome_intron_locations,
                row.start, row.end
            ),
            pbar
        ), axis=1
    ).values.tolist()
    training_sequence_frame = pd.DataFrame(training_sequences, columns=['input', 'target', 'position'])
    training_sequence_frame.loc[:, 'chromosome'] = chromosome
    training_sequence_frame.loc[:, 'type'] = index['type']
    pbar.close()
    return training_sequence_frame


@click.command()
@click.argument("assembly-path", type=Path)
def generate_sequences(assembly_path: Path):
    genes_path = assembly_path / "genes"
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    intron_locations_path = transcription_data_path / "intron_positions"
    index_path = transcription_data_path / "training_sequence_idx.csv"
    index = pd.read_csv(index_path)
    # Maybe split by chromosome and partially load intron locations, genes
    # intron_locations = get_intron_locations(intron_locations_path)
    tasks = [{
        "index": index[index.chromosome == c],
        "chromosome": c,
        "genes_path": genes_path,
        "intron_locations_path": intron_locations_path / f"chromosome-{c}.parquet"
    } for c in index.chromosome.unique()]
    pool = mp.Pool(processes=6)
    pbar = tqdm(total=len(tasks), ncols=80, desc="Generating")
    frame_write_path = transcription_data_path / "training_data.csv"
    header = True
    mode = "w+"
    try:
        for frame in pool.imap_unordered(get_chromosome_sequences_with_idx, tasks):
            frame.to_csv(frame_write_path, header=header, mode=mode, index=False)
            pbar.update(1)
            header = False
            mode = "a"
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    generate_sequences()
