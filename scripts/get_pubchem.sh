#!/usr/bin/env bash

data_dir="data"
pubchem_dir="data/pubchem_data"
pubchem_file="${pubchem_dir}/smiles.gz"

mkdir -p pubchem_dir

wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz -O pubchem_file

gzip -d pubchem_file
prefix="${dir}/${base}_part_"
pushd pubchem_dir
split -l 100000 "$file" "$prefix"
