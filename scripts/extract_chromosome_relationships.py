import click
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO

from llm_mito_scanner.data.transcription import get_gene_and_mrna_relationships, \
    write_mrna_gene_relationships


@click.command()
@click.argument("assembly_path", type=Path)
def extract(assembly_path: Path):
    mrna_to_gene_relationship_path = assembly_path / "relationships/mrna_to_gene"
    chromosomes_path = assembly_path / "chromosomes"
    chromosome_files = list(chromosomes_path.glob("*.gb"))
    if len(chromosome_files) == 0:
        print(f"No chromosomes found at {chromosomes_path.resolve()}")
    already_written_chromosome_relationships = [p.stem for p in mrna_to_gene_relationship_path.glob("*.csv")]
    chromosome_files = [p for p in chromosome_files if p.stem not in already_written_chromosome_relationships]
    if len(chromosome_files) == 0:
        print("All chromosome relationships extracted")
        return
    for chromosome_file in tqdm(chromosome_files, leave=True, ncols=80):
        if not (mrna_to_gene_relationship_path / f"{chromosome_file.stem}.csv").exists():
            with chromosome_file.open("rt") as f:
                chromosome = next(SeqIO.parse(f, "genbank"), None)
            chromosome_relationships = get_gene_and_mrna_relationships(chromosome)
            write_mrna_gene_relationships(chromosome_relationships, chromosome_file.stem, assembly_path)


if __name__ == "__main__":
    extract()
