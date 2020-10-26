import glob
import os
import re
import numpy as np

os.chdir('./')

brightness_train = []
brightness_val = []
contrast_train = []
contrast_val = []
saturation_train = []
saturation_val = []
hue_train = []
hue_val = []
rotation_train = []
rotation_val = []
translate_train = []
translate_val = []
scale_train = []
scale_val = []
shear_train = []
shear_val = []
vflip_train = []
vflip_val = []
hflip_train = []
hflip_val = []

for file in glob.glob("*.txt"):
    if 'brightness' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            brightness_train.append(train)
            brightness_val.append(val)

    if 'contrast' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            contrast_train.append(train)
            contrast_val.append(val)

    if 'saturation' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            saturation_train.append(train)
            saturation_val.append(val)

    if 'hue' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            hue_train.append(train)
            hue_val.append(val)

    if 'rotation' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            rotation_train.append(train)
            rotation_val.append(val)

    if 'translate' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            translate_train.append(train)
            translate_val.append(val)

    if 'scale' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            scale_train.append(train)
            scale_val.append(val)

    if 'shear' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            shear_train.append(train)
            shear_val.append(val)

    if 'vflip' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            vflip_train.append(train)
            vflip_val.append(val)

    if 'hflip' in file:
        with open(file) as f:
            content = f.readlines()
            train, val = re.findall(r"[-+]?\d*\.\d+|\d+", content[0])
            train, val = float(train), float(val)
            hflip_train.append(train)
            hflip_val.append(val)

brightness_train_std = np.std(np.array(brightness_train))
brightness_val_std = np.std(np.array(brightness_val))
contrast_train_std = np.std(np.array(contrast_train))
contrast_val_std = np.std(np.array(contrast_val))
saturation_train_std = np.std(np.array(saturation_train))
saturation_val_std = np.std(np.array(saturation_val))
hue_train_std = np.std(np.array(hue_train))
hue_val_std = np.std(np.array(hue_val))
rotation_train_std = np.std(np.array(rotation_train))
rotation_val_std = np.std(np.array(rotation_val))
translate_train_std = np.std(np.array(translate_train))
translate_val_std = np.std(np.array(translate_val))
scale_train_std = np.std(np.array(scale_train))
scale_val_std = np.std(np.array(scale_val))
shear_train_std = np.std(np.array(shear_train))
shear_val_std = np.std(np.array(shear_val))
vflip_train_std = np.std(np.array(vflip_train))
vflip_val_std = np.std(np.array(vflip_val))
hflip_train_std = np.std(np.array(hflip_train))
hflip_val_std = np.std(np.array(hflip_val))


brightness_train = np.mean(np.array(brightness_train))
brightness_val = np.mean(np.array(brightness_val))
contrast_train = np.mean(np.array(contrast_train))
contrast_val = np.mean(np.array(contrast_val))
saturation_train = np.mean(np.array(saturation_train))
saturation_val = np.mean(np.array(saturation_val))
hue_train = np.mean(np.array(hue_train))
hue_val = np.mean(np.array(hue_val))
rotation_train = np.mean(np.array(rotation_train))
rotation_val = np.mean(np.array(rotation_val))
translate_train = np.mean(np.array(translate_train))
translate_val = np.mean(np.array(translate_val))
scale_train = np.mean(np.array(scale_train))
scale_val = np.mean(np.array(scale_val))
shear_train = np.mean(np.array(shear_train))
shear_val = np.mean(np.array(shear_val))
vflip_train = np.mean(np.array(vflip_train))
vflip_val = np.mean(np.array(vflip_val))
hflip_train = np.mean(np.array(hflip_train))
hflip_val = np.mean(np.array(hflip_val))

print('mean')
print(brightness_train)
print(brightness_val)
print(contrast_train)
print(contrast_val)
print(saturation_train)
print(saturation_val)
print(hue_train)
print(hue_val)
print(rotation_train)
print(rotation_val)
print(translate_train)
print(translate_val)
print(scale_train)
print(scale_val)
print(shear_train)
print(shear_val)
print(vflip_train)
print(vflip_val)
print(hflip_train)
print(hflip_val)
print('std')
print(brightness_train_std)
print(brightness_val_std)
print(contrast_train_std)
print(contrast_val_std)
print(saturation_train_std)
print(saturation_val_std)
print(hue_train_std)
print(hue_val_std)
print(rotation_train_std)
print(rotation_val_std)
print(translate_train_std)
print(translate_val_std)
print(scale_train_std)
print(scale_val_std)
print(shear_train_std)
print(shear_val_std)
print(vflip_train_std)
print(vflip_val_std)
print(hflip_train_std)
print(hflip_val_std)

