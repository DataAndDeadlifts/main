import click
from pathlib import Path


from llm_mito_scanner.training.transcription.index import get_intron_locations
from llm_mito_scanner.training.transcription.generation import \
    get_mrna_locations, sample_sequences_idx


@click.command()
@click.argument("assembly-path", type=Path)
@click.argument("sample-number", type=int)
@click.option("--intron-proportion", type=float, default=0.25)
@click.option("--intron-edge-proportion", type=float, default=0.25)
@click.option("--mrna-proportion", type=float, default=0.25)
@click.option("--mrna-edge-proportion", type=float, default=0.25)
def generate_sample(
    assembly_path : Path, 
    sample_number: int, 
    intron_proportion: float, 
    intron_edge_proportion: float,
    mrna_proportion: float, 
    mrna_edge_proportion: float):
    ""
    training_data_path = assembly_path / "training"
    transcription_data_path = training_data_path / "transcription"
    intron_locations_path = transcription_data_path / "intron_positions"
    intron_locations = get_intron_locations(intron_locations_path)
    mrna_locations = get_mrna_locations(intron_locations)
    sample_idx = sample_sequences_idx(
        sample_number,
        intron_locations,
        mrna_locations,
        intron_prop=intron_proportion,
        intron_edge_prop=intron_edge_proportion,
        mrna_prop=mrna_proportion,
        mrna_edge_prop=mrna_edge_proportion,
    )
    sample_idx.to_csv(transcription_data_path / "training_sequence_idx.csv", index=False)


if __name__ == "__main__":
    generate_sample()
