#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/PACS
echo "Starting"
python fine_tune_alexnet_grey.py --test_domain art_painting --seed 0
python fine_tune_alexnet_grey.py --test_domain cartoon --seed 0
python fine_tune_alexnet_grey.py --test_domain photo --seed 0
python fine_tune_alexnet_grey.py --test_domain sketch --seed 0

python fine_tune_alexnet_grey.py --test_domain art_painting --seed 1
python fine_tune_alexnet_grey.py --test_domain cartoon --seed 1
python fine_tune_alexnet_grey.py --test_domain photo --seed 1
python fine_tune_alexnet_grey.py --test_domain sketch --seed 1

python fine_tune_alexnet_grey.py --test_domain art_painting --seed 2
python fine_tune_alexnet_grey.py --test_domain cartoon --seed 2
python fine_tune_alexnet_grey.py --test_domain photo --seed 2
python fine_tune_alexnet_grey.py --test_domain sketch --seed 2

python fine_tune_alexnet_grey.py --test_domain art_painting --seed 3
python fine_tune_alexnet_grey.py --test_domain cartoon --seed 3
python fine_tune_alexnet_grey.py --test_domain photo --seed 3
python fine_tune_alexnet_grey.py --test_domain sketch --seed 3

python fine_tune_alexnet_grey.py --test_domain art_painting --seed 4
python fine_tune_alexnet_grey.py --test_domain cartoon --seed 4
python fine_tune_alexnet_grey.py --test_domain photo --seed 4
python fine_tune_alexnet_grey.py --test_domain sketch --seed 4


echo "Done"
