#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/rotated_MNIST/augmentations
echo "Starting"
python experiment_augmentations_test_30.py --seed 0 --da flip
python experiment_augmentations_test_30.py --seed 1 --da flip
python experiment_augmentations_test_30.py --seed 2 --da flip
python experiment_augmentations_test_30.py --seed 3 --da flip
python experiment_augmentations_test_30.py --seed 4 --da flip
python experiment_augmentations_test_30.py --seed 5 --da flip
python experiment_augmentations_test_30.py --seed 6 --da flip
python experiment_augmentations_test_30.py --seed 7 --da flip
python experiment_augmentations_test_30.py --seed 8 --da flip
python experiment_augmentations_test_30.py --seed 9 --da flip

echo "Done"
