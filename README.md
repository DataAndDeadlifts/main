# Mito-Scanner

## Abstract

This project seeks to train a large language model on Human DNA sequences to identify snippets of DNA that are relevant to Mitochondrial function.

## Data Required

- The human genome ([link](https://www.ncbi.nlm.nih.gov/genome/guide/human/))
    - Acquired from: NIH NCBI
    - Format: Fasta
- RefSeq Reference Human Genome Annotation ([link](https://www.ncbi.nlm.nih.gov/genome/guide/human/))
    - Acquired from: NIH NCBI
    - Format: gff3

## Modeling Approach

- Model: TransformerModel from PyTorch.
- Criterion: Cross Entropy Loss
- Optimizer: SGD
- Schedule: StepLR
- Evaluation: Lowest Cross Entropy Loss
- Prediction: Apply to whole genome, return any sequences not currently associated with mitochondrial function

Notes:

- Apply nn.utils.clip_grad_norm_ to avoid exploding gradients

## Presentation

- Notebooks via nbdev
- Send results to Ekker, Clark lab

## Project Plan

1. Load, familiarize myself with Fasta, gff3
2. Build input vectors from genome, annotations
    1. Construct training vectors
        1. Break genome down into chunks for Transformer to handle
    2. Construct training labels
        1. Identify sequences related to mitochondrial function from annotations
3. Define model with PyTorch
    1. TransformerModel
4. Define training schedule with PyTorch
5. Train
6. Evaluate
7. Present