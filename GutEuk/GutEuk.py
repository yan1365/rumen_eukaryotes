#!/usr/bin/env python3

# Author: Ming Yan, The Ohio State University

import utils
import argparse
import gzip
import logging


# define args
description = '''\
GutEuk -- A deep-learning-based two-stage classifier to distinguish contigs/MAGs of prokaryotes, fungi or protozoa origin.

Designed specifically for gut microbiome.

In the first stage, the inputs are identified as either prokaryotes or eukaryotes origin (fungi or protozoa).

In the second stage, the eukaryotic sequences are further identified as either fungi or protozoa.

'''

Usage = 'Usage: GutEuk -i <input_file> -o <output_dir> [options]/ GutEuk -h.'

parser = argparse.ArgumentParser(description = description, usage = Usage)

parser.add_argument(
    "-i",
    "--input",
    metavar="input",
    help="The path to a fasta (or gzipped fasta) file.",
    type=str,
    required=True,
    nargs=1
)

parser.add_argument(
    "-o",
    "--output_dir",
    metavar="output",
    help="A path to output files.",
    default=None,
    required=True,
    nargs=1
)

parser.add_argument(
    "-m",
    "--min_len",
    help=f"""Minimum length of a sequence. Sequences shorter than min_len are discarded. 
    Default: 3000 bp.""",
    type=int,
    default=3000,
    nargs=1
)


parser.add_argument(
    "-t", "--threads", help="Number of threads used. The default is 1.", type=int, default=1, nargs=1
)

print(parser.description)
args = parser.parse_args()

# config variable
input_fasta = os.path.normpath(parser.input)
output_dir = os.path.normpath(parser.output_dir)
min_length = parser.min_len
threads = parser.threads
tmp_dir = os.path.normpath(f"{output_dir}/tmp")

# create a log file
logging.basicConfig(filename=os.path.join(output_dir, "log.txt")), level=logging.INFO, format='%(message)s')

if input_fasta.endswith(".gz"):
    bash_command = f"unzip {input_fasta} -d {tmp_dir}"
    subprocess.run(bash_command, shell=True, check=True, text=True)

# fasta to 








