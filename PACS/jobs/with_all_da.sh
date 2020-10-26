#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/PACS
echo "Starting"

python fine_tune_alexnet_with_all_da.py --seed 0 --test_domain art_painting
python fine_tune_alexnet_with_all_da.py --seed 1 --test_domain art_painting
python fine_tune_alexnet_with_all_da.py --seed 2 --test_domain art_painting
python fine_tune_alexnet_with_all_da.py --seed 3 --test_domain art_painting
python fine_tune_alexnet_with_all_da.py --seed 4 --test_domain art_painting

python fine_tune_alexnet_with_all_da.py --seed 0 --test_domain cartoon
python fine_tune_alexnet_with_all_da.py --seed 1 --test_domain cartoon
python fine_tune_alexnet_with_all_da.py --seed 2 --test_domain cartoon
python fine_tune_alexnet_with_all_da.py --seed 3 --test_domain cartoon
python fine_tune_alexnet_with_all_da.py --seed 4 --test_domain cartoon

python fine_tune_alexnet_with_all_da.py --seed 0 --test_domain photo
python fine_tune_alexnet_with_all_da.py --seed 1 --test_domain photo
python fine_tune_alexnet_with_all_da.py --seed 2 --test_domain photo
python fine_tune_alexnet_with_all_da.py --seed 3 --test_domain photo
python fine_tune_alexnet_with_all_da.py --seed 4 --test_domain photo

python fine_tune_alexnet_with_all_da.py --seed 0 --test_domain sketch
python fine_tune_alexnet_with_all_da.py --seed 1 --test_domain sketch
python fine_tune_alexnet_with_all_da.py --seed 2 --test_domain sketch
python fine_tune_alexnet_with_all_da.py --seed 3 --test_domain sketch
python fine_tune_alexnet_with_all_da.py --seed 4 --test_domain sketch

echo "Done"
