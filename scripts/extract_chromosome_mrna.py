import click
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool, current_process, Manager, Lock
from tqdm import tqdm
import sqlite3

warnings.simplefilter("ignore")

from llm_mito_scanner.data.transcription import get_genes
from llm_mito_scanner.training.transcription.generation import \
    get_mrna_from_gene, get_mrna_intron_locations, write_mrna


def update_pbar(result, pbar):
    pbar.update(1)
    return result


def extract_chromosome_mrna(args: dict):
    process_idx = next(iter(current_process()._identity), 0)
    chromosome = args.get("chromosome")
    assembly_path = args.get("assembly_path")
    batch_size = args.get('batch_size', 100)
    lock = args.get("lock")
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    mrna_data_path = transcription_data_path / "mrna"
    gene_lookup = get_genes(assembly_path, chromosome=chromosome).set_index("geneid").sequence.to_dict()
    intron_location_path = transcription_data_path / "intron_positions" / f"chromosome-{chromosome}.parquet"
    if not intron_location_path.exists():
        return
    intron_locations = pd.read_parquet(intron_location_path)
    intron_locations.loc[:, 'chromosome'] = chromosome
    chromosome_mrna = intron_locations[['chromosome', 'geneid', 'transcriptid', 'mrna_start', 'mrna_end']].drop_duplicates()
    # Remove already written mrna
    con = sqlite3.connect(assembly_path / "mrna.db")
    try:
        already_written_mrna = pd.read_sql_query("SELECT DISTINCT chromosome, geneid, transcriptid from mrna", con=con)
    except Exception as e:
        raise e
    finally:
        con.close()
    chromosome_mrna = chromosome_mrna.merge(
        already_written_mrna, 
        on=["chromosome", "geneid", "transcriptid"], 
        how="left", indicator=True)
    chromosome_mrna = chromosome_mrna[chromosome_mrna._merge == "left_only"]
    if chromosome_mrna.shape[0] == 0:
        return
    # Get mrna
    write_path = mrna_data_path / chromosome
    if not write_path.exists():
        write_path.mkdir()
    num_batches = max(1, chromosome_mrna.shape[0] / batch_size)
    chromosome_mrna = np.array_split(chromosome_mrna, num_batches)
    pbar = tqdm(total=len(chromosome_mrna), position=process_idx, desc=f"{chromosome}-generating", ncols=80, leave=False)
    for mrna in chromosome_mrna:
        mrna_sequences = mrna.apply(lambda row: get_mrna_from_gene(
            gene_lookup.get(row.geneid),
            row.mrna_start, row.mrna_end,
            get_mrna_intron_locations(
                row.chromosome, row.geneid, row.transcriptid, 
                intron_locations)), 
        axis=1)
        mrna_sequences.name = "sequence"
        mrna_sequences = pd.concat(
            [
                mrna,
                mrna_sequences
            ],axis=1)
        lock.acquire()
        try:
            write_mrna(assembly_path, chromosome, mrna_sequences)
        except Exception as e:
            raise e
        finally:
            lock.release()
        pbar.update(1)
    pbar.close()


@click.command()
@click.argument("assembly-path", type=Path)
@click.option("--batch-size", type=int, default=100)
def extract_mrna(assembly_path: Path, batch_size: int):
    con = sqlite3.connect(assembly_path / "genes.db")
    try:
        chromosomes = pd.read_sql_query(
            "SELECT DISTINCT chromosome from genes", con=con).chromosome.tolist()
    except Exception as e:
        raise e
    finally:
        con.close()
    manager = Manager()
    lock = manager.Lock()
    tasks = [{
        "chromosome": c,
        "assembly_path": assembly_path,
        "batch_size": batch_size,
        "lock": lock
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
