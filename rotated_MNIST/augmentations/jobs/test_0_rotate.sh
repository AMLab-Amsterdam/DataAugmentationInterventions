#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/rotated_MNIST/augmentations
echo "Starting"
python experiment_augmentations_test_0.py --seed 0 --da rotate
python experiment_augmentations_test_0.py --seed 1 --da rotate
python experiment_augmentations_test_0.py --seed 2 --da rotate
python experiment_augmentations_test_0.py --seed 3 --da rotate
python experiment_augmentations_test_0.py --seed 4 --da rotate
python experiment_augmentations_test_0.py --seed 5 --da rotate
python experiment_augmentations_test_0.py --seed 6 --da rotate
python experiment_augmentations_test_0.py --seed 7 --da rotate
python experiment_augmentations_test_0.py --seed 8 --da rotate
python experiment_augmentations_test_0.py --seed 9 --da rotate

echo "Done"
