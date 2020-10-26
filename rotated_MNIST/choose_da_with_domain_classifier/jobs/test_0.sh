#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/rotated_MNIST/choose_da_with_domain_classifier
echo "Starting"
python experiment_test_0.py --seed 0 --da brightness
python experiment_test_0.py --seed 0 --da contrast
python experiment_test_0.py --seed 0 --da saturation
python experiment_test_0.py --seed 0 --da hue
python experiment_test_0.py --seed 0 --da rotation
python experiment_test_0.py --seed 0 --da translate
python experiment_test_0.py --seed 0 --da scale
python experiment_test_0.py --seed 0 --da shear
python experiment_test_0.py --seed 0 --da none
python experiment_test_0.py --seed 0 --da hflip
python experiment_test_0.py --seed 0 --da vflip

python experiment_test_0.py --seed 1 --da brightness
python experiment_test_0.py --seed 1 --da contrast
python experiment_test_0.py --seed 1 --da saturation
python experiment_test_0.py --seed 1 --da hue
python experiment_test_0.py --seed 1 --da rotation
python experiment_test_0.py --seed 1 --da translate
python experiment_test_0.py --seed 1 --da scale
python experiment_test_0.py --seed 1 --da shear
python experiment_test_0.py --seed 1 --da none
python experiment_test_0.py --seed 1 --da hflip
python experiment_test_0.py --seed 1 --da vflip

python experiment_test_0.py --seed 2 --da brightness
python experiment_test_0.py --seed 2 --da contrast
python experiment_test_0.py --seed 2 --da saturation
python experiment_test_0.py --seed 2 --da hue
python experiment_test_0.py --seed 2 --da rotation
python experiment_test_0.py --seed 2 --da translate
python experiment_test_0.py --seed 2 --da scale
python experiment_test_0.py --seed 2 --da shear
python experiment_test_0.py --seed 2 --da none
python experiment_test_0.py --seed 2 --da hflip
python experiment_test_0.py --seed 2 --da vflip

python experiment_test_0.py --seed 3 --da brightness
python experiment_test_0.py --seed 3 --da contrast
python experiment_test_0.py --seed 3 --da saturation
python experiment_test_0.py --seed 3 --da hue
python experiment_test_0.py --seed 3 --da rotation
python experiment_test_0.py --seed 3 --da translate
python experiment_test_0.py --seed 3 --da scale
python experiment_test_0.py --seed 3 --da shear
python experiment_test_0.py --seed 3 --da none
python experiment_test_0.py --seed 3 --da hflip
python experiment_test_0.py --seed 3 --da vflip

python experiment_test_0.py --seed 4 --da brightness
python experiment_test_0.py --seed 4 --da contrast
python experiment_test_0.py --seed 4 --da saturation
python experiment_test_0.py --seed 4 --da hue
python experiment_test_0.py --seed 4 --da rotation
python experiment_test_0.py --seed 4 --da translate
python experiment_test_0.py --seed 4 --da scale
python experiment_test_0.py --seed 4 --da shear
python experiment_test_0.py --seed 4 --da none
python experiment_test_0.py --seed 4 --da hflip
python experiment_test_0.py --seed 4 --da vflip

echo "Done"
