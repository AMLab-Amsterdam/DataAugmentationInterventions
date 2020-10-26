#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/colored_mnist/
echo "Starting"
python main_with_all_da.py --seed 0
python main_with_all_da.py --seed 1
python main_with_all_da.py --seed 2
python main_with_all_da.py --seed 3
python main_with_all_da.py --seed 4
python main_with_all_da.py --seed 5
python main_with_all_da.py --seed 6
python main_with_all_da.py --seed 7
python main_with_all_da.py --seed 8
python main_with_all_da.py --seed 9


echo "Done"
