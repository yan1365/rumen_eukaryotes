# **GutEuk**

## Author information
01/01/2024  
Ming Yan  
The Ohio State University  
yan.1365@osu.edu
______

## Program description 
**GutEuk** -- A deep-learning-based two-stage classifier to distinguish contigs/MAGs of prokaryotes, fungi or protozoa origins.

- Designed specifically for gut metagenomes.

- In the first stage, the inputs are classified as either prokaryotes or eukaryotes origin (fungi or protozoa).

- In the second stage, the eukaryotic sequences are further classified as either fungi or protozoa.

- The classification of an input sequence is made by classifying each of the 5000 bp fragments of the input sequence (based on the majority rule).

- In the case when the votes from each class are equal (or lower than defined confidence levels), the input sequence is classified as "undetermined".

## Dependencies 
see ([environment.yaml](./environment.yaml))

 
## Installation
### Option 1
```bash
git clone git@github.com:yan1365/rumen_eukaryotes.git
cd GutEuk
conda env create -f environment.yaml -n GutEuk
conda activate GutEuk
```

Also consider to add script directory to $PATH by
```bash
export PATH="$MYPATH:$PATH" # Replace $MYPATH with the actual path where the 'GutEuk.py' is located.
source ~/.bashrc
```
Otherwise, provide full path when running the program.

## Usage 
### Basic usage
```bash
python GutEuk.py -i input.fasta -o output_dir -m 3000 -t 4 --to_fasta -s1 0.6 -s2 0.6 # contigs file as input
```
or
```bash
python GutEuk.py -b -i input_dir -o output_dir -m 3000 -t 4  -s1 0.6 -s2 0.6 # bins as input
```

### Program options
 - `-i input.fasta`, `--input input.fasta`: A path to the input fasta or gzipped fasta file (with a .gz suffix), or to a directory where bins are stored. --Required
 - `-o output_dir`, `--output_dir output_dir`: A path to store output files. Will be created if not exist. --Required
 - `-b`, `--bin`: Bins dir as input. --Optional
 - `-m MIN_LEN`, `--min_len MIN_LEN`: Minimum length of a sequence. Default: 3000. --Optional
 - `-s1 0.6`, `--stage1_confidence 0.6`: only give predictions when 60 percents of the input contig/bin fragments are classified as the same category in the stage 1. Default: 0.5. --Optional
 - `-s2 0.6`, `--stage2_confidence 0.6`: only give predictions when 60 percents of the input contig/bin fragments are classified as the same category in the stage 2. Default: 0.5. --Optional
 - `-t threads`, `--threads threads`: The number of threads to use. Default: 1. --Optional
 - `--to_fasta`: If provided, will output fasta files of each classes along with the final output to the output_dir. --Optional

### Test run (with testing datasets)
for contigs as input:
```bash
GutEuk.py -i test/contig_dir/test.fa -o ./  
```

for bins as input:
```bash
GutEuk.py -b -i test/bin_dir -o ./  
```




