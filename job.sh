#!/usr/bin/env bash
#SBATCH -J dfm
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=10G
#SBATCH -o none
#SBATCH -e none

# Run your script under your project
ssh bdcgpu01 "cd /mq/home/m1cak00/code/hmc-estimation/; julia --project=. --threads=auto models/dfm.jl"