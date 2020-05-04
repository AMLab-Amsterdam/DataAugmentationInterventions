#!/bin/sh
#SBATCH --job-name=anneal_beta
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH -t 8:00:00

cd ~/domain_gen/TwoLatentSpacesVAE/paper_experiments/ablation_on_rotated_MNIST/partition_conditional_prior_aux_classifier/

source activate twoVAE

echo "Starting"
python experiment_anneal_betas.py &> ~/anneal_betas_seed_0.log&

wait # Waits for parallel jobs to finish
echo "Done"
