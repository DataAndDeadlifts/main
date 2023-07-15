# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/04 training.transcription.sampling.ipynb.

# %% auto 0
__all__ = ['sample_intron_edges', 'sample_introns', 'sample_mrna', 'sample_mrna_edges', 'sample_sequences_idx',
           'make_mrna_file_index', 'get_mrna_file_index', 'make_gene_sequence_lookup', 'make_mrna_sequence_lookup',
           'get_mrna_partitions', 'get_training_sequences_with_idx', 'get_index_sequences',
           'get_chromosome_idx_sequences']

# %% ../../../nbs/04 training.transcription.sampling.ipynb 5
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import random
import warnings
import time
from datetime import timedelta
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

from ...data.download import load_config, \
    get_latest_assembly_path, get_genomic_genbank_path
from ...data.transcription import get_chromosome_genes
from .index import get_intron_locations
from .generation import get_mrna_locations, get_gene

# %% ../../../nbs/04 training.transcription.sampling.ipynb 16
def sample_intron_edges(
        locations: pd.DataFrame, n: int, 
        random_state: int = 42, offset: int = -32, length: int = 64) -> pd.DataFrame:
    "Get training instances where either the start of end of an intron is in the center of the sequence."
    start_n = int(n / 2)
    end_n = n - start_n
    replace = True if (start_n > locations.shape[0]) | (end_n > locations.shape[0]) else False
    starts = locations.sample(start_n, replace=replace, random_state=random_state)
    ends = locations.sample(end_n, replace=replace, random_state=random_state)
    frames = []
    for f, slice_origin in zip([starts, ends], ['intron_start', 'intron_end']):
        f_slice_start = (f[slice_origin] - f.mrna_start + offset).apply(lambda val: max(0, val))
        f.loc[:, 'mrna_len'] = f.mrna_end - f.mrna_start
        f.loc[:, 'start'] = f_slice_start
        f.loc[:, 'end'] = (f_slice_start + length)
        f.loc[:, 'end'] = f[['end', 'mrna_len']].min(axis=1)
        # Detect short sequences
        short_mask = (f.end - f.start) != length
        short_end_mask = short_mask & (f.end == f.mrna_len)
        short_start_mask = short_mask & (f.start == 0)
        if short_end_mask.sum() > 0:
            # Fix the samples with short ends
            f_short_ends = f.loc[short_end_mask, :]
            f_short_lengths = f_short_ends.end - f_short_ends.start
            f_short_adjustments = f_short_lengths - length
            f_short_ends.loc[:, 'start'] = f_short_ends.start + f_short_adjustments
            f.loc[short_end_mask, :] = f_short_ends
        if short_start_mask.sum() > 0:
            # Fix the samples with short starts
            f_short_starts = f.loc[short_start_mask, :]
            f_short_lengths = f_short_starts.end - f_short_starts.start
            f_short_adjustments = (f_short_lengths - length).mul(-1)
            f_short_starts.loc[:, 'end'] = f_short_starts.end + f_short_adjustments
            f.loc[short_start_mask, :] = f_short_starts
        f = f[['chromosome', 'geneid', 'transcriptid', 'start', 'end']]
        frames.append(f)
    intron_edges = pd.concat(frames, axis=0)
    intron_edges.loc[:, 'type'] = 'intron-edge'
    return intron_edges

# %% ../../../nbs/04 training.transcription.sampling.ipynb 22
def sample_introns(
        locations: pd.DataFrame, n: int,
        random_state: int = 42, length: int = 64) -> pd.DataFrame:
    random.seed(random_state)
    "Get training instances where most of the tokens are <intron>."
    replace = False if n < locations.shape[0] else True
    intron_sample = locations.sample(n, replace=replace, random_state=random_state)
    # Handle sequences of varying sizes
    intron_sample.loc[:, 'intron_length'] = intron_sample.intron_end - intron_sample.intron_start
    intron_len_mask = intron_sample.intron_length <= length
    small_introns = intron_sample[intron_len_mask]
    large_introns = intron_sample[~intron_len_mask]
    sample_frames = []
    if small_introns.shape[0] > 0:
        # For those introns less than length, center, return the whole thing
        # Start at intron start
        small_intron_slice_center = small_introns.intron_start
        # Shift slice center half the distance of the target sequence
        small_intron_slice_center = small_intron_slice_center.subtract(int(length / 2)).apply(lambda val: max(0, val))
        small_introns.loc[:, 'start'] = small_intron_slice_center
        small_introns.loc[:, 'end'] = small_introns.start + length
        small_introns.loc[:, 'end'] = small_introns[['end', 'mrna_end']].min(axis=1)
        small_introns = small_introns[['chromosome', 'geneid', 'transcriptid', 'start', 'end']]
        small_introns.loc[:, 'type'] = 'intron-small'
        sample_frames.append(small_introns)
    if large_introns.shape[0] > 0:
        # For larger introns, identify the range we can slice to avoid edges
        large_introns.loc[:, 'slice_max'] = large_introns.intron_end - length
        large_introns.loc[:, 'slice_range'] = large_introns.apply(lambda row: range(row.intron_start, row.slice_max + 1, 1), axis=1)
        large_introns.loc[:, 'start'] = large_introns.slice_range.apply(lambda r: random.choice(r))
        large_introns.loc[:, 'end'] = large_introns.start + length
        large_introns = large_introns[['chromosome', 'geneid', 'transcriptid', 'start', 'end']]
        large_introns.loc[:, 'type'] = 'intron'
        sample_frames.append(large_introns)
    # Randomly select a slice point within the identified range
    introns = pd.concat(sample_frames, axis=0)
    return introns

# %% ../../../nbs/04 training.transcription.sampling.ipynb 27
def sample_mrna(
        mrna_locations: pd.DataFrame, n: int, 
        random_state: int = 42, length: int = 64) -> pd.DataFrame:
    "Get a sample or mrna sequence locations"
    replace = False if n < mrna_locations.shape[0] else True
    mrna_locations = mrna_locations.sample(n, replace=replace, random_state=random_state)
    # For small mrna sections, do the same thing we did with the introns
    # Handle sequences of varying sizes
    mrna_locations.loc[:, 'length'] = mrna_locations.end - mrna_locations.start
    mrna_len_mask = mrna_locations.length <= length
    small_sequences = mrna_locations[mrna_len_mask]
    large_sequences = mrna_locations[~mrna_len_mask]
    sample_frames = []
    if small_sequences.shape[0] > 0:
        # For those introns less than length, center, return the whole thing
        # Start at intron start
        small_sequences_slice_center = small_sequences.start
        # Shift slice center half the distance of the target sequence
        small_sequences_slice_center = small_sequences_slice_center.subtract(int(length / 2)).apply(lambda val: max(0, val))
        small_sequences.loc[:, 'start'] = small_sequences_slice_center
        small_sequences.loc[:, 'end'] = small_sequences.start + length
        small_sequences.loc[:, 'end'] = small_sequences[['end', 'mrna_end']].min(axis=1)
        small_sequences = small_sequences[['chromosome', 'geneid', 'transcriptid', 'start', 'end']]
        small_sequences.loc[:, 'type'] = 'mrna-small'
        sample_frames.append(small_sequences)
    if large_sequences.shape[0] > 0:
        # For larger introns, identify the range we can slice to avoid edges
        large_sequences.loc[:, 'slice_max'] = large_sequences.end - length
        large_sequences.loc[:, 'slice_range'] = large_sequences.apply(lambda row: range(row.start, row.slice_max + 1, 1), axis=1)
        large_sequences.loc[:, 'start'] = large_sequences.slice_range.apply(lambda r: random.choice(r))
        large_sequences.loc[:, 'end'] = large_sequences.start + length
        large_sequences = large_sequences[['chromosome', 'geneid', 'transcriptid', 'start', 'end']]
        large_sequences.loc[:, 'type'] = 'mrna'
        sample_frames.append(large_sequences)
    # Randomly select a slice point within the identified range
    return pd.concat(sample_frames, axis=0, ignore_index=True)

# %% ../../../nbs/04 training.transcription.sampling.ipynb 30
def sample_mrna_edges(locations: pd.DataFrame, n: int, random_state: int = 42, length: int = 64) -> pd.DataFrame:
    "Get the beginning and end of mrna"
    locations = locations.drop_duplicates(
        ['chromosome', 'geneid', 'transcriptid', 'mrna_start', 'mrna_end']
    ).drop(['intron_start', 'intron_end'], axis=1).reset_index(drop=True)
    n_start = int(n / 2)
    n_end = n - n_start
    replace = False if (n_start < locations.shape[0]) or (n_end < locations.shape[0]) else True
    mrna_starts = locations.sample(
        n_start, replace=replace, random_state=random_state
        ).rename({'mrna_start': 'start'}, axis=1)
    mrna_starts.loc[:, 'end'] = mrna_starts.start + length
    mrna_starts.loc[:, 'end'] = mrna_starts[['mrna_end', 'end']].min(axis=1)
    mrna_starts.drop(['mrna_end'], axis=1, inplace=True)
    mrna_ends = locations.sample(
        n_end, replace=replace, random_state=random_state).rename({'mrna_end': 'end'}, axis=1)
    mrna_ends.loc[:, 'start'] = mrna_ends.end - length
    mrna_ends.loc[:, 'start'] = mrna_ends[['mrna_start', 'start']].max(axis=1)
    mrna_ends.drop(['mrna_start'], axis=1, inplace=True)
    sample_edges = pd.concat([mrna_starts, mrna_ends], axis=0, ignore_index=True)
    sample_edges = sample_edges[['chromosome', 'geneid', 'transcriptid', 'start', 'end']]
    sample_edges.loc[:, 'type'] = 'mrna-edge'
    return sample_edges

# %% ../../../nbs/04 training.transcription.sampling.ipynb 32
def sample_sequences_idx(
        n: int, 
        intron_locations: pd.DataFrame,
        mrna_locations: pd.DataFrame,
        intron_prop: float, intron_edge_prop: float, 
        mrna_prop: float, mrna_edge_prop: float,
        random_state: int = 42,
        length: int = 64) -> pd.DataFrame:
    "Build training dataset from intron locations."
    intron_sample = sample_introns(
        intron_locations, int(n * intron_prop), 
        random_state=random_state, length=length)
    intron_edge_sample = sample_intron_edges(
        intron_locations, int(n * intron_edge_prop), 
        random_state=random_state, length=length)
    mrna_sample = sample_mrna(mrna_locations, int(n * mrna_prop), 
        random_state=random_state, length=length)
    mrna_edge_sample = sample_mrna_edges(intron_locations, int(n * mrna_edge_prop),
        random_state=random_state, length=length)
    sample = pd.concat([
        intron_sample,
        intron_edge_sample,
        mrna_sample,
        mrna_edge_sample
    ], axis=0, ignore_index=True)
    return sample

# %% ../../../nbs/04 training.transcription.sampling.ipynb 35
def make_mrna_file_index(mrna_path: Path) -> pd.DataFrame:
    ""
    parquet_files = list(mrna_path.glob("*/*.parquet"))
    index_frames = []
    for p in parquet_files:
        p_chromosome = p.parent.name
        p_frame = pd.read_parquet(p, columns=['geneid', 'transcriptid'])
        p_frame.loc[:, 'chromosome'] = p_chromosome
        p_frame.loc[:, 'path'] = p
        index_frames.append(p_frame)
    return pd.concat(index_frames, axis=0, ignore_index=True)


def get_mrna_file_index(transcription_path: Path) -> dict[tuple[str, str, str], Path]:
    "Get the file a particular mRNA resides in."
    mrna_index_path = transcription_path / "mrna_index.csv"
    if not mrna_index_path.exists():
        mrna_index = make_mrna_file_index(transcription_path / "mrna")
        mrna_index.to_csv(mrna_index_path)
    else:
        mrna_index = pd.read_csv(mrna_index_path)
    mrna_index.loc[:, 'path'] = mrna_index.path.apply(Path)
    return mrna_index.set_index(['chromosome', 'geneid', 'transcriptid']).path.to_dict()

# %% ../../../nbs/04 training.transcription.sampling.ipynb 40
def make_gene_sequence_lookup(
        genes: pd.DataFrame) -> dict[tuple[str, str], str]:
    return genes.set_index(['chromosome', 'geneid']).sequence.to_dict()


def make_mrna_sequence_lookup(
        mrna: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    return mrna.set_index(['chromosome', 'geneid', 'transcriptid']).mrna.to_dict()

# %% ../../../nbs/04 training.transcription.sampling.ipynb 44
def get_mrna_partitions(paths: list[Path], transcript_ids: list[str] = None) -> pd.DataFrame:
    frames = []
    for p in paths:
        p_frame = pd.read_parquet(p)
        if isinstance(transcript_ids, list):
            p_frame = p_frame[p_frame.transcriptid.isin(transcript_ids)]
            if p_frame.shape[0] == 0:
                continue
        p_chromosome = p.parent.name
        p_frame.loc[:, 'chromosome'] = p_chromosome
        frames.append(p_frame)
    return pd.concat(frames, axis=0, ignore_index=True)

# %% ../../../nbs/04 training.transcription.sampling.ipynb 48
def get_training_sequences_with_idx(
        gene: str, mrna: str,
        start: int, end: int,
        ) -> tuple[str, str]:
    ""
    return ",".join(list(gene)[start: end]), ",".join(mrna.split(",")[start: end])

# %% ../../../nbs/04 training.transcription.sampling.ipynb 55
def get_index_sequences(
        index: pd.DataFrame,
        assembly_path: Path,
        chromosome: str = None) -> pd.DataFrame:
    # Make gene lookup
    gene_lookup = get_chromosome_genes(
        assembly_path, 
        chromosome=chromosome, 
        gene_ids=index.geneid.unique().tolist())
    gene_lookup = make_gene_sequence_lookup(gene_lookup)
    # Make mRNA lookup
    mrna_lookup = get_mrna_partitions(
        index.mrna_partition.unique().tolist(),
        index.transcriptid.unique().tolist())
    mrna_lookup = make_mrna_sequence_lookup(mrna_lookup)
    sequences = index.apply(
        lambda row: get_training_sequences_with_idx(
            gene_lookup.get((row.chromosome, row.geneid)),
            mrna_lookup.get((row.chromosome, row.geneid, row.transcriptid)),
            row.start, row.end
        ), axis=1).values.tolist()
    sequences = pd.DataFrame(sequences, columns=['input', 'target'])
    return pd.concat(
        [
            index,
            sequences
        ], axis=1, ignore_index=False)

# %% ../../../nbs/04 training.transcription.sampling.ipynb 61
def get_chromosome_idx_sequences(args: dict):
    chromosome = args.get("chromosome")
    index = args.get('index')
    assembly_path = args.get("assembly")
    pbar_position = args.get("position", 1)
    chunk_size = args.get("chunk_size", 50)
    write_size = args.get("write_size", 10000)
    save = args.get("save", False)
    index.sort_values(["transcriptid", "mrna_partition"], inplace=True)
    write_path = assembly_path / "training/transcription/sequences" / chromosome
    if not write_path.exists() and save:
        write_path.mkdir(parents=True)
    num_chunks = max(1, index.shape[0] / chunk_size)
    index_chunks = []
    curr_chunk = None
    for p in index.mrna_partition.unique():
        p_index = index[index.mrna_partition == p]
        if p_index.shape[0] > chunk_size:
            num_chunks = max(1, p_index.shape[0] / chunk_size)
            p_index_chunks = np.array_split(p_index, num_chunks)
            for p_chunk in p_index_chunks:
                if curr_chunk is None:
                    curr_chunk = p_chunk
                else:
                    curr_chunk = pd.concat([curr_chunk, p_chunk], axis=0, ignore_index=True)
                    if curr_chunk.shape[0] >= chunk_size:
                        index_chunks.append(curr_chunk)
                        curr_chunk = None
            index_chunks.extend(p_index_chunks)
        else:
            if curr_chunk is None:
                curr_chunk = p_index
            else:
                curr_chunk = pd.concat([curr_chunk, p_index], axis=0, ignore_index=True)
                if curr_chunk.shape[0] >= chunk_size:
                    index_chunks.append(curr_chunk)
                    curr_chunk = None
    if curr_chunk is not None:
        index_chunks.append(curr_chunk)
    pbar = tqdm(
        total=len(index_chunks), 
        position=pbar_position, leave=False, ncols=80, desc=f"{chromosome}")
    batch_counter = 1
    sequences = []
    for chunk in index_chunks:
        chromosome_index_chunk_sequences = get_index_sequences(
            chunk,
            assembly_path,
            chromosome=chromosome
        )
        chromosome_index_chunk_sequences = chromosome_index_chunk_sequences[[
            "chromosome", "geneid", "transcriptid",
            "start", "end",
            "type",
            "input", "target"
        ]]
        sequences.append(chromosome_index_chunk_sequences)
        if save and sum([f.shape[0] for f in sequences]) >= write_size:
            write_path_chunk = write_path / f"partition-{str(batch_counter).zfill(3)}.parquet"
            pd.concat(sequences, axis=0, ignore_index=True).to_parquet(write_path_chunk, index=False)
            sequences = []
            batch_counter += 1
        pbar.update(1)
    if save and len(sequences) > 0:
        write_path_chunk = write_path / f"partition-{str(batch_counter).zfill(3)}.parquet"
        pd.concat(sequences, axis=0, ignore_index=True).to_parquet(write_path_chunk, index=False)
    pbar.close()
