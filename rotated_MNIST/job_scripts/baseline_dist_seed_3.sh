#!/bin/sh
#SBATCH --job-name=baseline_dist_seed_3
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH -t 08:00:00

cd ~/domain_gen/TwoLatentSpacesVAE/paper_experiments/ablation_on_rotated_MNIST/baseline/

source activate twoVAE

echo "Starting"
python experiment_baseline_dist.py --seed 3 &> ~/baseline_dist_seed_3.log&

wait # Waits for parallel jobs to finish
echo "Done"
