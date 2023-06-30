import os
from pathlib import Path
from multiprocessing import Pool, current_process
import click
from Bio import SeqIO
from tqdm import tqdm

from llm_mito_scanner.data.transcription import get_chromosome_gene_info, \
    write_chromosome_gene_info


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
    genes_path = assembly_path / "genes"
    already_written_chromosomes = [p.stem for p in genes_path.glob("*.csv")]
    chromosome_files = [p for p in chromosome_files if p.stem not in already_written_chromosomes]
    if len(chromosome_files) == 0:
        print(f"All chromosome files processed")
        return
    pool = Pool(os.cpu_count() - 1)
    pbar = tqdm(total=len(chromosome_files), desc="Overall", ncols=80, leave=True, position=0)
    try:
        for (path, frame) in pool.imap_unordered(get_and_write_all_chromosome_gene_info_wrapper, chromosome_files):
            write_chromosome_gene_info(assembly_path, path.stem, frame)
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    extract()
