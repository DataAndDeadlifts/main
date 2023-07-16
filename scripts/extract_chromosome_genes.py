import os
from pathlib import Path
from multiprocessing import Pool, current_process
import click
import warnings
from tqdm import tqdm
import sqlite3
import pandas as pd

warnings.simplefilter("ignore")
from Bio import SeqIO

from llm_mito_scanner.data.transcription import get_chromosome_gene_info, \
    write_genes


threads = 8
processes = os.cpu_count() - 1


def get_and_write_all_chromosome_gene_info_wrapper(path: Path):
    with path.open("rt") as f:
        chromosome = next(SeqIO.parse(f, "genbank"), None)
    pbar_position = next(iter(current_process()._identity), 0)
    chromosome_gene_info = get_chromosome_gene_info(chromosome, pbar_position=pbar_position)
    return path, chromosome_gene_info


@click.command()
@click.argument("assembly_path", type=Path)
def extract(assembly_path: Path):
    chromosomes_path = assembly_path / "chromosomes"
    chromosome_files = list(chromosomes_path.glob("*.gb"))
    if len(chromosome_files) == 0:
        print(f"No files to process at path {chromosomes_path.resolve()}")
        return
    con = sqlite3.connect(assembly_path / "genes.db")
    try:
        written_chromosomes = pd.read_sql_query("SELECT DISTINCT chromosome from genes", con=con).chromosome.tolist()
    except Exception as e:
        raise e
    finally:
        con.close()
    chromosome_files = [p for p in chromosome_files if p.stem not in written_chromosomes]
    if len(chromosome_files) == 0:
        print(f"All chromosome files processed")
        return
    pool = Pool(6)
    pbar = tqdm(total=len(chromosome_files), desc="Overall", ncols=80, leave=True, position=0)
    try:
        for (path, frame) in pool.imap_unordered(get_and_write_all_chromosome_gene_info_wrapper, chromosome_files):
            if frame.shape[0] > 0:
                write_genes(assembly_path, path.stem, frame)
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    extract()
