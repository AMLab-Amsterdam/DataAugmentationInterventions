#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/synthetic_data
echo "Starting"
python main_IRM.py --setup_hidden 0 --setup_hetero 0 --setup_scramble 0 > IRM_FOU.txt
python main_IRM.py --setup_hidden 0 --setup_hetero 1 --setup_scramble 0 > IRM_FEU.txt
python main_IRM.py --setup_hidden 1 --setup_hetero 0 --setup_scramble 0 > IRM_POU.txt
python main_IRM.py --setup_hidden 1 --setup_hetero 1 --setup_scramble 0 > IRM_PEU.txt

echo "Done"
