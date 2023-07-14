# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01 data.transcription.ipynb.

# %% auto 0
__all__ = ['filter_chromosome_features_by_type', 'get_feature_qualifiers', 'get_feature_dbxrefs', 'get_feature_dbxref_xref',
           'get_feature_geneid', 'get_feature_transcript_id', 'get_chromosome_gene_info', 'write_chromosome_gene_info',
           'get_gene_and_mrna_relationships', 'write_mrna_gene_relationships', 'get_mrna_gene_id',
           'get_gene_seq_record', 'normalize_mrna_positions', 'get_mrna_bookends', 'extract_sequence_with_positions',
           'get_mrna_intron_positions', 'make_intron_position_dataframe']

# %% ../../nbs/01 data.transcription.ipynb 7
import warnings
warnings.simplefilter("ignore")
import os
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, SimpleLocation, Seq, CompoundLocation, Location
from tqdm.auto import tqdm
import pandas as pd
import typing
from tqdm import tqdm
from multiprocessing import current_process
from copy import deepcopy

tqdm.pandas()

from .download import load_config, \
    get_latest_assembly_path, get_genomic_genbank_path

# %% ../../nbs/01 data.transcription.ipynb 13
def filter_chromosome_features_by_type(
        chromosome_record: SeqRecord, 
        feature_type: str,
        ) -> list[tuple[int, SeqFeature]]:
    return [(i, f) for i, f in enumerate(chromosome_record.features) if f.type == feature_type]

# %% ../../nbs/01 data.transcription.ipynb 14
def get_feature_qualifiers(feature: SeqFeature) -> typing.Dict[str, typing.Any]:
    return getattr(feature, "qualifiers", None)


def get_feature_dbxrefs(feature: SeqFeature) -> str | None:
    feature_qualifiers = get_feature_qualifiers(feature)
    if feature_qualifiers is None:
        return None
    feature_dbxrefs = feature_qualifiers.get("db_xref", None)
    return feature_dbxrefs


def get_feature_dbxref_xref(feature: SeqFeature, prefix: str) -> str | None:
    feature_dbxrefs = get_feature_dbxrefs(feature)
    if feature_dbxrefs is None:
        return None
    tag_db_xref = next(iter([x for x in feature_dbxrefs if x.startswith(prefix)]), None)
    return tag_db_xref


def get_feature_geneid(feature: SeqFeature) -> str | None:
    return get_feature_dbxref_xref(feature, "GeneID")


def get_feature_transcript_id(feature: SeqFeature) -> str | None:
    return next(iter(get_feature_qualifiers(feature).get("transcript_id", [])), None)

# %% ../../nbs/01 data.transcription.ipynb 23
def get_chromosome_gene_info(
        chromosome_record: SeqRecord,
        pbar_position: int = 0
        ) -> pd.DataFrame:
    chromosome_genes = [t[1] for t in filter_chromosome_features_by_type(chromosome_record, "gene")]
    chromosome_gene_ids = list(map(get_feature_geneid, chromosome_genes))
    chromosome_gene_sequences = list(map(
        lambda seq_feature: str(seq_feature.extract(chromosome_record).seq),
        tqdm(chromosome_genes, leave=False, ncols=80, position=pbar_position, desc=f"Process-{pbar_position}")
    ))
    pos_strand_positions = list(map(lambda f: f.location.start, chromosome_genes))
    neg_strand_positions = list(map(lambda f: f.location.end, chromosome_genes))
    gene_df = pd.DataFrame(
        chromosome_gene_ids, columns=['geneid']
    )
    gene_df.loc[:, 'sequence'] = chromosome_gene_sequences
    gene_df.loc[:, 'pos_strand_position'] = pos_strand_positions
    gene_df.loc[:, 'neg_strand_position'] = neg_strand_positions
    return gene_df
    

# %% ../../nbs/01 data.transcription.ipynb 26
def write_chromosome_gene_info(assembly_path: Path, chromosome_tag: str, frame: pd.DataFrame):
    genes_path = assembly_path / "genes"
    if not genes_path.exists():
        genes_path.mkdir()
    gene_info_path = genes_path / f"{chromosome_tag}.csv"
    frame.to_csv(gene_info_path, index=False)

# %% ../../nbs/01 data.transcription.ipynb 30
def get_gene_and_mrna_relationships(
        chromosome: SeqRecord,
        ) -> pd.DataFrame:
    ""
    mrna_features = filter_chromosome_features_by_type(chromosome, "mRNA")
    relationships = set()
    for idx, mrna in mrna_features:
        mrna_transcript_id = get_feature_transcript_id(mrna)
        mrna_gene_id = get_feature_geneid(mrna)
        if mrna_gene_id is not None and mrna_transcript_id is not None:
            relationship_tuple = (mrna_gene_id, mrna_transcript_id, idx)
            relationships.add(relationship_tuple)

    chromosome_relationships_df = pd.DataFrame(
        relationships, 
        columns=['geneid', 'transcript_id', 'transcript_feature_idx']
    ).drop_duplicates(subset=["geneid", "transcript_id"]).sort_values("transcript_feature_idx", ascending=True)

    gene_idx = pd.DataFrame(
        [(idx, get_feature_geneid(f)) for idx, f in filter_chromosome_features_by_type(chromosome, "gene")],
        columns=["gene_feature_idx", "geneid"]
    )

    chromosome_relationships_df = chromosome_relationships_df.merge(
        gene_idx, on=['geneid']
    ).drop_duplicates(
        subset=['geneid', 'transcript_id']
        ).sort_values("gene_feature_idx", ascending=True)
    return chromosome_relationships_df

# %% ../../nbs/01 data.transcription.ipynb 34
def write_mrna_gene_relationships(relationships: pd.DataFrame, chromosome: str, assembly_path: Path):
    relationship_path = assembly_path / "relationships"
    mrna_to_gene_path = relationship_path / "mrna_to_gene"
    if not mrna_to_gene_path.exists():
        mrna_to_gene_path.mkdir(parents=True)
    chromosome_relationship_path = mrna_to_gene_path / f"{chromosome}.csv"
    relationships.to_csv(chromosome_relationship_path, index=False)

# %% ../../nbs/01 data.transcription.ipynb 41
def get_mrna_gene_id(mrna_tup: tuple[int, SeqFeature], relationships: pd.DataFrame):
    idx, mrna = mrna_tup
    mrna_transcript_id = get_feature_transcript_id(mrna)
    mrna_gene_id = relationships[relationships.transcript_id == mrna_transcript_id]
    if mrna_gene_id.shape[0] == 0:
        return None
    return mrna_gene_id.iloc[0, :].geneid

# %% ../../nbs/01 data.transcription.ipynb 43
def get_gene_seq_record(gene_id: str, genes: pd.DataFrame) -> tuple[tuple[int, int], SeqRecord]:
    ""
    gene_id_row = genes[genes.geneid == gene_id]
    if gene_id_row.shape[0] == 0:
        return None
    gene_id_row = gene_id_row.iloc[0, :]
    gene_id_seqrecord = SeqRecord(Seq(gene_id_row.sequence))
    return gene_id_row.pos_strand_position, gene_id_row.neg_strand_position, gene_id_seqrecord

# %% ../../nbs/01 data.transcription.ipynb 49
def normalize_mrna_positions(
        mrna_tup: tuple[int, SeqFeature], 
        gene_record_tup: tuple[tuple[int, int], SeqRecord],
        debug: bool = False
        ) -> list[tuple[int, int]]:
    idx, mrna = mrna_tup
    pos_strand_position, neg_strand_position, gene_record = gene_record_tup
    is_neg_strand = mrna.location.parts[0].strand == -1
    if not is_neg_strand:
        norm_positions = [p - pos_strand_position for p in mrna.location.parts]
    else:
        norm_positions = [p - neg_strand_position for p in mrna.location.parts]
        inverted_positions = []
        for p in norm_positions:
            inverted_p = SimpleLocation(abs(p.end),abs(p.start), strand=1)
            inverted_positions.append(inverted_p)
        norm_positions = inverted_positions
    norm_position_ints = [(int(p.start), int(p.end)) for p in norm_positions]
    return norm_position_ints

# %% ../../nbs/01 data.transcription.ipynb 51
def get_mrna_bookends(
        mrna_tup: tuple[int, SeqFeature], 
        gene_record_tup: tuple[tuple[int, int], SeqRecord]) -> tuple[int, int]:
    norm_mrna_positions = normalize_mrna_positions(mrna_tup, gene_record_tup)
    start = norm_mrna_positions[0][0]
    end = norm_mrna_positions[-1][-1]
    return start, end

# %% ../../nbs/01 data.transcription.ipynb 55
def extract_sequence_with_positions(positions: list[tuple[int, int]], sequence: str):
    sequence_extracted_list = []
    for start, end in positions:
        position_sequence = sequence[start: end]
        sequence_extracted_list.append(position_sequence)
    return "".join(sequence_extracted_list)

# %% ../../nbs/01 data.transcription.ipynb 64
def get_mrna_intron_positions(
        mrna_tup: tuple[int, SeqFeature],
        gene_record_tup: tuple[tuple[int, int], SeqRecord],
) -> list[tuple[int, int]]:
    "Get intron positions to replace in the input sequence."
    mrna_norm_positions = normalize_mrna_positions(mrna_tup, gene_record_tup)
    # Get the starting gene location of the mrna
    mrna_start, mrna_end = get_mrna_bookends(mrna_tup, gene_record_tup)
    # Get the first end of the spliced transcript
    prev_end = None
    intron_positions = []
    for pos_start, pos_end in mrna_norm_positions:
        if prev_end is None:
            prev_end = pos_end
            continue
        intron = [prev_end, pos_start]
        intron_positions.append(intron)
        prev_end = int(pos_end)
    intron_positions = [(p[0] - mrna_start, p[1] - mrna_start) for p in intron_positions]
    return intron_positions

# %% ../../nbs/01 data.transcription.ipynb 69
def make_intron_position_dataframe(
        gene_ids: list[str], 
        transcript_ids: list[str], 
        mrna_bookends: list[tuple[int, int]],
        intron_positions: list[tuple[int, int]]):
    frame = pd.DataFrame(
        gene_ids,
        columns=['geneid']
    )
    frame.loc[:, 'transcriptid'] = transcript_ids
    frame.loc[:, 'intron_position'] = intron_positions
    frame.loc[:, 'bookends'] = mrna_bookends
    frame = frame.explode('intron_position').reset_index(drop=True)
    frame.dropna(subset=['intron_position'], inplace=True)
    if frame.shape[0] > 0:
        frame.loc[:, 'intron_start'] = frame.intron_position.str[0]
        frame.loc[:, 'intron_end'] = frame.intron_position.str[1]
        frame.drop('intron_position', axis=1, inplace=True)
        frame.loc[:, 'mrna_start'] = frame.bookends.str[0]
        frame.loc[:, 'mrna_end'] = frame.bookends.str[1]
        frame.drop('bookends', axis=1, inplace=True)
    else:
        frame = None
    return frame