import os
from pathlib import Path
import typing
from multiprocessing import current_process
from multiprocessing.pool import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
from Bio import SeqIO, SeqRecord, SeqFeature
import pandas as pd
from tqdm import tqdm

from llm_mito_scanner.data.download import get_latest_assembly_path
from llm_mito_scanner.data.transcription import get_gene_and_mrna_features, \
    get_gene_and_mrna_relationships, process_gene_mrna_pair, \
        get_feature_transcript_id, get_feature_geneid


threads = 8
processes = os.cpu_count() - 1


def process_gene_mrna_pair_wrapper(
        chromosome: SeqRecord,
        gene: SeqFeature,
        mrna: SeqFeature
):
    # Make pandas series that denotes a training instance
    pair_series = process_gene_mrna_pair(
        chromosome,
        gene, mrna
    )
    # Annotate gene id, mrna transcript id
    pair_series.at['geneid'] = get_feature_geneid(gene)
    pair_series.at['mrna_transcript_id'] = get_feature_transcript_id(mrna)
    return pair_series


def process_chromosome_features(chromosome_path: Path) -> typing.Tuple[Path, pd.DataFrame]:
    global threads
    "Process features in an assembly's chromosome genbank file."
    with chromosome_path.open("rt") as f:
        chromosome = next(SeqIO.parse(f, "genbank"), None)
    # For each chromosome genbank file, in it's own process
    #   Get gene, mrna features
    gene_features, mrna_features = get_gene_and_mrna_features(chromosome)
    #   Identify gene, mrna relationships
    mrna_to_protein_relationships = get_gene_and_mrna_relationships(
       gene_features, mrna_features
    )
    #   For each relationship, in a threadpool
    #       Make pandas series that denotes a training instance
    pbar_position = current_process()._identity[0]
    task_pbar = tqdm(
        total=len(mrna_to_protein_relationships), 
        position=pbar_position, 
        ncols=80, 
        leave=False,
        desc=f"Process-{pbar_position}"
        )
    pair_series_list = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        tasks = {
            executor.submit(
                process_gene_mrna_pair_wrapper, 
                chromosome, 
                gene_features[idx[1]], 
                mrna_features[idx[0]]
            ): idx for idx in mrna_to_protein_relationships
        }
        for future in as_completed(tasks):
            pair_series = future.result()
            pair_series_list.append(pair_series)
            task_pbar.update(1)
    #   Combine training instances
    training_instances = pd.DataFrame(pair_series_list).sort_values(
        'geneid', ascending=True).reset_index(drop=True)
    task_pbar.close()
    return chromosome_path, training_instances
    

@click.command()
@click.argument("data_path", type=Path)
@click.option("--assembly", type=str, default=None)
def process_transcription_data(data_path: Path, assembly: str):
    "Extract features from the genbank files, format for transcription training."
    global processes, threads
    data_path_raw = data_path / "raw"
    data_path_raw_assemblies = data_path_raw / "assemblies"
    if not data_path_raw_assemblies.exists():
        raise FileNotFoundError(
            "Can't find assemblies path at " + \
                f"{data_path_raw_assemblies.resolve()} " + \
                    f"given input {data_path.resolve()}"
            )
    # Get the latest assembly file
    if assembly is not None:
        assembly_path = data_path_raw_assemblies / assembly
    else:
        assembly_path = get_latest_assembly_path(data_path_raw_assemblies)
    assembly_transcription_training_data_path = assembly_path / "training/transcription"
    if not assembly_transcription_training_data_path.exists():
        assembly_transcription_training_data_path.mkdir(parents=True)
    chromosome_path = assembly_path / "chromosomes"
    if not chromosome_path.exists():
        raise FileNotFoundError("Couldn't find extracted chromosomes at "+\
                                f"{chromosome_path.resolve()}")
    # Index chromosome genbank files
    chromosome_files = list(chromosome_path.glob("*.gb"))
    if len(chromosome_files) == 0:
        raise FileNotFoundError("No genbank chromosome files found at " + \
                                f"{chromosome_path.resolve()}")
    # For each chromosome genbank file, get its training instances
    pbar = tqdm(total=len(chromosome_files), ncols=80, leave=True, desc="Overall")
    process_pool = Pool(processes=processes)
    try:
        for source_chromosome_path, training_data in \
            process_pool.imap_unordered(
                process_chromosome_features,
                chromosome_files
                ):
            chromosome_transcription_training_data_path = \
                assembly_transcription_training_data_path / f"{source_chromosome_path.stem}.parquet"
            training_data.to_parquet(chromosome_transcription_training_data_path, index=False)
            pbar.update(1)
    except Exception as e:
        raise e
    finally:
        process_pool.close()
        pbar.close()


if __name__ == "__main__":
    process_transcription_data()
