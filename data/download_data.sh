#!/usr/bin/env bash

files=(
acute_myeloid_obs.tsv.gz
acute_myeloid_pcs_harmonized.tsv.gz
acute_myeloid_pcs.tsv.gz
ircolitis_blood_cd8_obs.tsv.gz
ircolitis_blood_cd8_pcs_harmonized.tsv.gz
ircolitis_blood_cd8_pcs.tsv.gz
lisi_lisi.tsv.gz
lisi_metadata.tsv.gz
lisi_x.tsv.gz
pbmc_3500_meta.tsv.gz
pbmc_3500_pcs_harmonized.tsv.gz
pbmc_3500_pcs.tsv.gz
)

for f in ${files[*]}
do
  wget -N https://immunogenomics.io/downloads/$f
done
