import click
from pathlib import Path
import typing
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from multiprocessing import current_process, Pool


from llm_mito_scanner.data.transcription import get_input_sequence, \
    get_target_sequence, filter_chromosome_features_by_type, \
    get_mrna_gene_id, get_feature_transcript_id, get_gene_seq_record, \
    get_sequence_bytes


def write_training_data_partition(
        write_dir: Path,
        training_data: list[tuple[str, str, str, str]]
        ):
    ""
    dataframe = pd.DataFrame(
        training_data,
        columns=['geneid', 'transcriptid', 'input', 'target']
    )
    if not write_dir.exists():
        write_dir.mkdir(parents=True)
    write_path = write_dir / "sequences.parquet"
    dataframe.to_parquet(write_path)


def make_chromosome_transcription_training_data(args: dict[str, typing.Any]):
    process_idx = next(iter(current_process()._identity), 0)
    tqdm_args = {"position": process_idx, "ncols": 80, "leave": False}
    assembly_path = args.get("assembly")
    chromosome_path = args.get("chromosome")
    genes_path = args.get("genes")
    partition_size_limit_mb = args.get("partition_size")
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    sequence_path = transcription_data_path / "sequences"
    tqdm_desc_base = f"{chromosome_path.stem}-"
    # Read the chromosome genes
    chromosome_genes = pd.read_csv(genes_path)
    # Read the chromosome
    with chromosome_path.open("rt") as f:
        chromosome = next(SeqIO.parse(f, "genbank"), None)
    # Get the mRNA
    chromosome_mrna = filter_chromosome_features_by_type(chromosome, "mRNA")
    # Get the relationships
    relationships_path = assembly_path / "relationships"/ "mrna_to_gene" / f"{chromosome_path.stem}.csv"
    relationships = pd.read_csv(relationships_path)
    # Get the mRNA gene ids
    mrna_geneids = list(map(
        lambda m: get_mrna_gene_id(
            m, relationships
        ), 
        tqdm(chromosome_mrna, 
             desc=tqdm_desc_base + "gene-ids", **tqdm_args)
    ))
    mrna_transcriptids = list(map(
        lambda m: get_feature_transcript_id(
            m[1]
        ),
        tqdm(chromosome_mrna, 
             desc=tqdm_desc_base + "transcript-ids", **tqdm_args)
    ))
    chromosome_sequence_path = sequence_path / f"chromosome={chromosome_path.stem}"
    training_data_pbar = tqdm(total=len(mrna_geneids), desc=tqdm_desc_base+"transforming", **tqdm_args)
    partition_data = []
    partition_size = 0
    partition = 0
    for gene_id, transcript_id, mrna_tup in zip(mrna_geneids, mrna_transcriptids, chromosome_mrna):
        # Make the gene sequence record
        gene_record_tup = get_gene_seq_record(gene_id, chromosome_genes)
        input_sequence = get_input_sequence(mrna_tup, gene_record_tup)
        target_sequence = get_target_sequence(mrna_tup, gene_record_tup, input_sequence)
        # Format for writing
        input_sequence_str = ",".join(input_sequence)
        target_sequence_str = ",".join(target_sequence)
        partition_data.append((gene_id, transcript_id, input_sequence_str, target_sequence_str))
        partition_size += get_sequence_bytes(input_sequence_str, target_sequence_str)
        training_data_pbar.update(1)
        if partition_size >= partition_size_limit_mb:
            partition_path = chromosome_sequence_path / f"partition={partition}"
            write_training_data_partition(partition_path, partition_data)
            partition_data = []
            partition_size = 0
            partition += 1
    if len(partition_data) > 0:
        partition_path = chromosome_sequence_path / f"partition={partition}"
        write_training_data_partition(partition_path, partition_data)
    training_data_pbar.close()


@click.command()
@click.argument("assembly_path", type=Path)
@click.option("--partition-size-mb", type=int, default=500)
def make_transcription_training_data(assembly_path: Path, partition_size_mb: int):
    # Collect available chromosomes to extract data from
    chromosomes_path = assembly_path / "chromosomes"
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    sequence_path = transcription_data_path / "sequences"
    sequence_path_list = list(sequence_path.glob("*.csv"))
    sequence_path_stems = [p.stem for p in sequence_path_list]
    # Make args for multiprocessing pool
    tasks = [
        {
            "assembly": assembly_path,
            "chromosome": p,
            "genes": assembly_path / "genes" / f"{p.stem}.csv",
            "partition_size": (1000 ** 2) * partition_size_mb
        } for p in chromosomes_path.glob("*.gb") if p.stem not in sequence_path_stems
    ]
    if len(tasks) == 0:
        print("All chromosomes processed")
        return
    pbar = tqdm(total=len(tasks), desc="Overall", ncols=80, leave=True, position=0)
    pool = Pool(4)
    try:
        for res in pool.imap_unordered(make_chromosome_transcription_training_data, tasks):
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    make_transcription_training_data()
