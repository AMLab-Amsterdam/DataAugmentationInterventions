#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/PACS
echo "Starting"
python fine_tune_alexnet_domain_classifier.py --da none --seed 0
python fine_tune_alexnet_domain_classifier.py --da jitter --seed 0

python fine_tune_alexnet_domain_classifier.py --da none --seed 1
python fine_tune_alexnet_domain_classifier.py --da jitter --seed 1

python fine_tune_alexnet_domain_classifier.py --da none --seed 2
python fine_tune_alexnet_domain_classifier.py --da jitter --seed 2

python fine_tune_alexnet_domain_classifier.py --da none --seed 3
python fine_tune_alexnet_domain_classifier.py --da jitter --seed 3

python fine_tune_alexnet_domain_classifier.py --da none --seed 4
python fine_tune_alexnet_domain_classifier.py --da jitter --seed 4

echo "Done"
