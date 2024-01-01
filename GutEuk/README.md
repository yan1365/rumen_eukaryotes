# **GutEuk**

## Author information
01/01/2024  
Ming Yan  
The Ohio State University  
yan.1365@osu.edu
______

## Program description 
**GutEuk** -- A deep-learning-based two-stage classifier to distinguish contigs/MAGs of prokaryotes, fungi or protozoa origins.

- Designed specifically for gut microbiome.

- In the first stage, the inputs are classified as either prokaryotes or eukaryotes origin (fungi or protozoa).

- In the second stage, the eukaryotic sequences are further classified as either fungi or protozoa.

- The classification of an input sequence is made by classifying each of the 5000 bp fragments of the input sequence (based on the majority rule).

- In the case when the votes from each class are equal, the input sequence is classified as "undetermined".

## Dependencies and installation 
To do############################

## Usage 
### Test run (on a test dataset)
```bash
wget url for test.fa #############################
python GutEuk.py -i test.fa -o ./
```

### Basic usage
```bash
python GutEuk.py -i input.fasta -o output_dir -m 3000 -t 4 --to_fasta
```

### Program options
 - `-i input.fasta`, `--input input.fasta`: A path to the input fasta or gzipped fasta file (with a .gz suffix). --Required
 - `-o output_dir`, `--output_dir output_dir`: A path to store output files. Will be created if not exist. --Required
 - `-m MIN_LEN`, `--min_len MIN_LEN`: Minimum length of a sequence. Default: 3000. --Optional
 - `-t threads`, `--threads threads`: The number of threads to use. Default: 1. --Optional
 - `--to_fasta`: If provided, will output fasta files of each classes along with the final output to the output_dir. --Optional


