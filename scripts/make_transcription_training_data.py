import click
from pathlib import Path
import typing
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from multiprocessing import current_process, Pool


from llm_mito_scanner.data.transcription import \
    filter_chromosome_features_by_type, \
    get_mrna_gene_id, get_feature_transcript_id, \
    get_mrna_bookends, get_mrna_intron_positions, \
    get_gene_seq_record, make_intron_position_dataframe


def write_training_data_partition(
        write_path: Path,
        training_data: pd.DataFrame
        ):
    ""
    if not write_path.parent.exists():
        write_path.parent.mkdir(parents=True)
    training_data.to_parquet(write_path)


def make_chromosome_transcription_training_data(args: dict[str, typing.Any]):
    process_idx = next(iter(current_process()._identity), 0)
    tqdm_args = {"position": process_idx, "ncols": 80, "leave": False}
    # Parse args
    assembly_path = args.get("assembly")
    chromosome_path = args.get("chromosome")
    genes_path = args.get("genes")
    # Setup paths
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    intron_path = transcription_data_path / "intron_positions"
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
    training_data_pbar = tqdm(total=len(mrna_geneids), desc=tqdm_desc_base+"transforming", **tqdm_args)
    all_gene_ids = []
    all_transcript_ids = []
    all_mrna_bookends = []
    all_intron_positions = []
    for gene_id, transcript_id, mrna_tup in zip(mrna_geneids, mrna_transcriptids, chromosome_mrna):
        # Make the gene sequence record
        gene_record_tup = get_gene_seq_record(gene_id, chromosome_genes)
        mrna_bookends = get_mrna_bookends(mrna_tup, gene_record_tup)
        transcript_intron_positions = get_mrna_intron_positions(mrna_tup, gene_record_tup)
        # Format for writing
        all_gene_ids.append(gene_id)
        all_transcript_ids.append(transcript_id)
        all_mrna_bookends.append(mrna_bookends)
        all_intron_positions.append(transcript_intron_positions)
        training_data_pbar.update(1)
    partition_path = intron_path / f"chromosome-{chromosome_path.stem}.parquet"
    partition_df = make_intron_position_dataframe(
        all_gene_ids, all_transcript_ids, all_mrna_bookends, all_intron_positions
    )
    if partition_df is not None:
        write_training_data_partition(partition_path, partition_df)
    training_data_pbar.close()


@click.command()
@click.argument("assembly_path", type=Path)
def make_transcription_training_data(assembly_path: Path):
    # Collect available chromosomes to extract data from
    chromosomes_path = assembly_path / "chromosomes"
    # Make args for multiprocessing pool
    tasks = [
        {
            "assembly": assembly_path,
            "chromosome": p,
            "genes": assembly_path / "genes" / f"{p.stem}.csv",
        } for p in chromosomes_path.glob("*.gb")
    ]
    if len(tasks) == 0:
        print("All chromosomes processed")
        return
    pbar = tqdm(total=len(tasks), desc="Overall", ncols=80, leave=True, position=0)
    pool = Pool(4)
    try:
        for _ in pool.imap_unordered(make_chromosome_transcription_training_data, tasks):
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pbar.close()


if __name__ == "__main__":
    make_transcription_training_data()
