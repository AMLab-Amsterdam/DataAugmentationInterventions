#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/rotated_MNIST/augmentations
echo "Starting"
python experiment_augmentations_test_0_all_da.py --seed 0
python experiment_augmentations_test_0_all_da.py --seed 1
python experiment_augmentations_test_0_all_da.py --seed 2
python experiment_augmentations_test_0_all_da.py --seed 3
python experiment_augmentations_test_0_all_da.py --seed 4
python experiment_augmentations_test_0_all_da.py --seed 5
python experiment_augmentations_test_0_all_da.py --seed 6
python experiment_augmentations_test_0_all_da.py --seed 7
python experiment_augmentations_test_0_all_da.py --seed 8
python experiment_augmentations_test_0_all_da.py --seed 9

echo "Done"
