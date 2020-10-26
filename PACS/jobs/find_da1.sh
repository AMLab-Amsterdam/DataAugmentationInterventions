#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/PACS
echo "Starting"

python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da brightness
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da contrast
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da saturation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da hue
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da rotation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da translate
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da scale
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da shear
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da vflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da hflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain art_painting --da none

python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da brightness
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da contrast
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da saturation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da hue
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da rotation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da translate
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da scale
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da shear
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da vflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da hflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain cartoon --da none

python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da brightness
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da contrast
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da saturation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da hue
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da rotation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da translate
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da scale
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da shear
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da vflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da hflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain photo --da none

python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da brightness
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da contrast
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da saturation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da hue
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da rotation
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da translate
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da scale
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da shear
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da vflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da hflip
python fine_tune_alexnet_find_da.py --seed 1 --test_domain sketch --da none


echo "Done"
