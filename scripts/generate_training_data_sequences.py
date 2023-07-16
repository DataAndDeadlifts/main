import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
from multiprocessing import Pool

warnings.simplefilter("ignore")

tqdm.pandas()

from llm_mito_scanner.training.transcription.sampling import get_gene_transcript_samples_wrapper, get_mrna_file_index


@click.command()
@click.argument("assembly-path", type=Path)
@click.option("--batch-size", type=int, default=100000)
def generate_sequences(assembly_path: Path, batch_size: int):
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    index_path = transcription_data_path / "training_sequence_idx.csv"
    index = pd.read_csv(index_path)
    mrna_file_index = get_mrna_file_index(transcription_data_path)
    index.loc[:, 'mrna_partition'] = index.apply(
        lambda row: mrna_file_index.get((row.chromosome, row.geneid, row.transcriptid)),
        axis=1)
    del mrna_file_index
    index.loc[:, "idx"] = pd.Series(zip(index.start, index.end))
    index = index.groupby(["chromosome", "geneid", "transcriptid", "mrna_partition"]).agg({
        "idx": list,
        "type": list
    })
    index.loc[:, 'samples'] = index.idx.apply(len)
    index.sort_values('samples', ascending=False, inplace=True)
    index.reset_index(drop=False, inplace=True)
    tasks = [
        {
            "chromosome": row.chromosome,
            "geneid": row.geneid,
            "transcriptid": row.transcriptid,
            "index": row.idx,
            "types": row['type'],
            "assembly_path": assembly_path,
            "partition_path": row.mrna_partition
        } for _, row in index.iterrows()
    ]
    del index
    partition_path = transcription_data_path / "sequences"
    if not partition_path.exists():
        partition_path.mkdir()
    pool = Pool(1)
    task_pbar = tqdm(tasks, ncols=80, desc="Generating", leave=False)
    try:
        batch_count = 1
        frame_list_row_count = 0
        frame_list = []
        for frame in pool.imap_unordered(get_gene_transcript_samples_wrapper, tasks):
            # Write frame
            frame_list.append(frame)
            frame_list_row_count += frame.shape[0]
            task_pbar.update(1)
            if frame_list_row_count >= batch_size:
                write_path = partition_path / f"batch-{batch_count}.parquet"
                pd.concat(frame_list, axis=0, ignore_index=True).to_parquet(write_path, index=False)
                batch_count += 1
                frame_list = []
                frame_list_row_count = 0
        if len(frame_list) > 0:
            write_path = partition_path / f"batch-{batch_count}.parquet"
            pd.concat(frame_list, axis=0, ignore_index=True).to_parquet(write_path, index=False)
    except Exception as e:
        raise e
    finally:
        pool.close()
        task_pbar.close()
    


if __name__ == "__main__":
    generate_sequences()
