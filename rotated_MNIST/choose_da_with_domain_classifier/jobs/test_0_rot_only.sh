#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/rotated_MNIST/choose_da_with_domain_classifier
echo "Starting"
python experiment_test_0_only_rotation.py --seed 0 --da rotation1
python experiment_test_0_only_rotation.py --seed 0 --da rotation2
python experiment_test_0_only_rotation.py --seed 0 --da rotation3
python experiment_test_0_only_rotation.py --seed 0 --da rotation4
python experiment_test_0_only_rotation.py --seed 0 --da rotation5

python experiment_test_0_only_rotation.py --seed 1 --da rotation1
python experiment_test_0_only_rotation.py --seed 1 --da rotation2
python experiment_test_0_only_rotation.py --seed 1 --da rotation3
python experiment_test_0_only_rotation.py --seed 1 --da rotation4
python experiment_test_0_only_rotation.py --seed 1 --da rotation5

python experiment_test_0_only_rotation.py --seed 2 --da rotation1
python experiment_test_0_only_rotation.py --seed 2 --da rotation2
python experiment_test_0_only_rotation.py --seed 2 --da rotation3
python experiment_test_0_only_rotation.py --seed 2 --da rotation4
python experiment_test_0_only_rotation.py --seed 2 --da rotation5

python experiment_test_0_only_rotation.py --seed 3 --da rotation1
python experiment_test_0_only_rotation.py --seed 3 --da rotation2
python experiment_test_0_only_rotation.py --seed 3 --da rotation3
python experiment_test_0_only_rotation.py --seed 3 --da rotation4
python experiment_test_0_only_rotation.py --seed 3 --da rotation5

python experiment_test_0_only_rotation.py --seed 4 --da rotation1
python experiment_test_0_only_rotation.py --seed 4 --da rotation2
python experiment_test_0_only_rotation.py --seed 4 --da rotation3
python experiment_test_0_only_rotation.py --seed 4 --da rotation4
python experiment_test_0_only_rotation.py --seed 4 --da rotation5


echo "Done"
