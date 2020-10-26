#!/bin/sh
#SBATCH --job-name=baseline
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH -t 08:00:00

cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/rotated_MNIST/domain_classifier

source activate twoVAE

echo "Starting"
python domain_classifier.py --da none --seed 0
python domain_classifier.py --da none --seed 1
python domain_classifier.py --da none --seed 2
python domain_classifier.py --da none --seed 3
python domain_classifier.py --da none --seed 4
python domain_classifier.py --da none --seed 5
python domain_classifier.py --da none --seed 6
python domain_classifier.py --da none --seed 7
python domain_classifier.py --da none --seed 8
python domain_classifier.py --da none --seed 9

wait # Waits for parallel jobs to finish
echo "Done"
