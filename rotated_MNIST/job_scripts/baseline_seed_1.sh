#!/bin/sh
#SBATCH --job-name=baseline_seed_1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH -t 08:00:00

cd ~/domain_gen/TwoLatentSpacesVAE/paper_experiments/ablation_on_rotated_MNIST/baseline/

source activate twoVAE

echo "Starting"
python experiment_baseline.py --seed 1 &> ~/baseline_seed_1.log&

wait # Waits for parallel jobs to finish
echo "Done"
