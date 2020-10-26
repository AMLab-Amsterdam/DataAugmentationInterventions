#!/bin/bash
cd ~/AdditionalDriveA/DeployedProjects/TwoLatentSpacesVAE/paper_experiments/colored_mnist/
echo "Starting"
#python choose_da_with_domain_classifier.py --seed 0 --da brightness
python choose_da_with_domain_classifier.py --seed 0 --da contrast
python choose_da_with_domain_classifier.py --seed 0 --da saturation
#python choose_da_with_domain_classifier.py --seed 0 --da hue
#python choose_da_with_domain_classifier.py --seed 0 --da rotation
#python choose_da_with_domain_classifier.py --seed 0 --da translate
#python choose_da_with_domain_classifier.py --seed 0 --da scale
#python choose_da_with_domain_classifier.py --seed 0 --da shear
#python choose_da_with_domain_classifier.py --seed 0 --da hflip
#python choose_da_with_domain_classifier.py --seed 0 --da vflip
#python choose_da_with_domain_classifier.py --seed 0 --da none

#python choose_da_with_domain_classifier.py --seed 1 --da brightness
python choose_da_with_domain_classifier.py --seed 1 --da contrast
python choose_da_with_domain_classifier.py --seed 1 --da saturation
#python choose_da_with_domain_classifier.py --seed 1 --da hue
#python choose_da_with_domain_classifier.py --seed 1 --da rotation
#python choose_da_with_domain_classifier.py --seed 1 --da translate
#python choose_da_with_domain_classifier.py --seed 1 --da scale
#python choose_da_with_domain_classifier.py --seed 1 --da shear
#python choose_da_with_domain_classifier.py --seed 1 --da hflip
#python choose_da_with_domain_classifier.py --seed 1 --da vflip
#python choose_da_with_domain_classifier.py --seed 1 --da none

#python choose_da_with_domain_classifier.py --seed 2 --da brightness
python choose_da_with_domain_classifier.py --seed 2 --da contrast
python choose_da_with_domain_classifier.py --seed 2 --da saturation
#python choose_da_with_domain_classifier.py --seed 2 --da hue
#python choose_da_with_domain_classifier.py --seed 2 --da rotation
#python choose_da_with_domain_classifier.py --seed 2 --da translate
#python choose_da_with_domain_classifier.py --seed 2 --da scale
#python choose_da_with_domain_classifier.py --seed 2 --da shear
#python choose_da_with_domain_classifier.py --seed 2 --da hflip
#python choose_da_with_domain_classifier.py --seed 2 --da vflip
#python choose_da_with_domain_classifier.py --seed 2 --da none

#python choose_da_with_domain_classifier.py --seed 3 --da brightness
python choose_da_with_domain_classifier.py --seed 3 --da contrast
python choose_da_with_domain_classifier.py --seed 3 --da saturation
#python choose_da_with_domain_classifier.py --seed 3 --da hue
#python choose_da_with_domain_classifier.py --seed 3 --da rotation
#python choose_da_with_domain_classifier.py --seed 3 --da translate
#python choose_da_with_domain_classifier.py --seed 3 --da scale
#python choose_da_with_domain_classifier.py --seed 3 --da shear
#python choose_da_with_domain_classifier.py --seed 3 --da hflip
#python choose_da_with_domain_classifier.py --seed 3 --da vflip
#python choose_da_with_domain_classifier.py --seed 3 --da none

#python choose_da_with_domain_classifier.py --seed 4 --da brightness
python choose_da_with_domain_classifier.py --seed 4 --da contrast
python choose_da_with_domain_classifier.py --seed 4 --da saturation
#python choose_da_with_domain_classifier.py --seed 4 --da hue
#python choose_da_with_domain_classifier.py --seed 4 --da rotation
#python choose_da_with_domain_classifier.py --seed 4 --da translate
#python choose_da_with_domain_classifier.py --seed 4 --da scale
#python choose_da_with_domain_classifier.py --seed 4 --da shear
#python choose_da_with_domain_classifier.py --seed 4 --da hflip
#python choose_da_with_domain_classifier.py --seed 4 --da vflip
#python choose_da_with_domain_classifier.py --seed 4 --da none


echo "Done"
