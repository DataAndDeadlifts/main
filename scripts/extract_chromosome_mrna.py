import click
from pathlib import Path
import pandas as pd
import warnings
from multiprocessing import Pool, current_process
from tqdm import tqdm

warnings.simplefilter("ignore")

from llm_mito_scanner.data.transcription import read_all_chromosome_gene_info
from llm_mito_scanner.training.transcription.generation import get_mrna_from_gene, get_intron_locations, get_mrna_intron_locations


def update_pbar(result, pbar):
    pbar.update(1)
    return result


def extract_chromosome_mrna(args: dict):
    process_idx = next(iter(current_process()._identity), 0)
    chromosome = args.get("chromosome")
    assembly_path = args.get("assembly_path")
    batch_size = args.get('batch_size', 500)
    gene_path = assembly_path / "genes"
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    mrna_data_path = transcription_data_path / "mrna"
    chromosome_genes = pd.read_csv(gene_path / f"{chromosome}.csv").set_index("geneid").sequence.to_dict()
    chromosome_intron_locations = pd.read_parquet(transcription_data_path / "intron_positions" / f"chromosome-{chromosome}.parquet")
    chromosome_intron_locations.loc[:, 'chromosome'] = chromosome
    chromosome_mrna = chromosome_intron_locations[['geneid', 'transcriptid', 'mrna_start', 'mrna_end']].drop_duplicates()
    # Get mrna
    pbar = tqdm(total=chromosome_mrna.shape[0], position=process_idx, desc=f"{chromosome}-generating", ncols=80, leave=False)
    write_path = mrna_data_path / chromosome
    if not write_path.exists():
        write_path.mkdir()
    batch_counter = 1
    row_batch = []
    for i, (_, row) in enumerate(chromosome_mrna.iterrows()):
        row_copy = row.copy()
        _, mrna_sequence = get_mrna_from_gene(
            chromosome_genes.get(row.geneid),
            row.mrna_start, row.mrna_end,
            get_mrna_intron_locations(chromosome, row.geneid, row.transcriptid, chromosome_intron_locations))
        mrna_sequence = ",".join(mrna_sequence)
        # row_copy.loc["gene"] = gene_sequence # We can get this elsewhere, lets save some disc space
        row_copy.loc["mrna"] = mrna_sequence
        row_batch.append(row_copy)
        if len(row_batch) >= batch_size:
            write_path_batch = write_path / f"partition-{str(batch_counter).zfill(3)}.parquet"
            pd.DataFrame(row_batch).to_parquet(write_path_batch, index=False)
            row_batch = []
            batch_counter += 1
        if i % 10 == 0:
            pbar.update(10)
    if len(row_batch) > 0:
        write_path_batch = write_path / f"partition-{str(batch_counter).zfill(3)}.parquet"
        pd.DataFrame(row_batch).to_parquet(write_path_batch, index=False)
    pbar.update(chromosome_mrna.shape[0] - (chromosome_mrna.shape[0] // 10))
    pbar.close()


@click.command()
@click.argument("assembly-path", type=Path)
def extract_mrna(assembly_path: Path):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    mrna_data_path = transcription_data_path / "mrna"
    if not mrna_data_path.exists():
        mrna_data_path.mkdir()
    genes_path = assembly_path / "genes"
    gene_files = list(genes_path.glob("*.csv"))
    chromosomes = [p.stem for p in gene_files]
    tasks = [{
        "chromosome": c,
        "assembly_path": assembly_path
    } for c in chromosomes]
    task_pbar = tqdm(total=len(tasks), ncols=80, desc="Extracting", leave=False)
    pool = Pool(6)
    try:
        for _ in pool.imap_unordered(extract_chromosome_mrna, tasks):
            task_pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        task_pbar.close()


if __name__ == "__main__":
    extract_mrna()
