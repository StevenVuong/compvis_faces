"""
This is meant to convert files in "rgb_mask" to "int_mask"
"""
import os
import numpy as np
import operator
from PIL import Image

# paths to current masks
rgb_image_mask_path = "./rgb_mask/masks"
rgb_image_colormap_path = "./rgb_mask/colormap.txt"
int_colormask_save_path = "./int_mask/colormap.csv"

# load colormap codes as array
colormap_array = np.loadtxt(rgb_image_colormap_path, dtype=str)

# get a set of string and integer keys
string_keys = [c.split(':')[0] for c in colormap_array]
int_keys = [idx[0] for idx in enumerate(colormap_array)]

# get the rgb mappings
rgb_values = [str([int(num) for num in c.split(':')[1].split(',')]).replace(' ', '')\
               for c in colormap_array]

# create rgb, int mapping dictionary
mapping_dict = {k:v for k, v in zip(rgb_values, int_keys)}
# sort by value
mapping_dict = dict(sorted(mapping_dict.items(), key=operator.itemgetter(1)))

# create integer mapping dict to string
int_mapping_dict = {idx:string for idx, string in enumerate(string_keys)}

# create save directory to int mask
if not os.path.exists("./int_mask/"): os.makedirs("./int_mask/")
if not os.path.exists("./int_mask/masks/"): os.makedirs("./int_mask/masks/")
# save int-colormap as csv
with open(int_colormask_save_path,"w") as f:
    for key in int_mapping_dict.keys():
        f.write("%s, %s\n" % (key, int_mapping_dict[key]))

# modify RGB mask to int mask if not exist
for idx, rgb_mask in enumerate(os.listdir(rgb_image_mask_path)): # loop through 
    print("Processing mask:",rgb_mask)
    rgb_mask_path = os.path.join(rgb_image_mask_path, rgb_mask)
    rgb_mask_image = np.asarray(Image.open(rgb_mask_path))

    num_rows = rgb_mask_image.shape[0]
    num_cols = rgb_mask_image.shape[1]
    output_array = np.zeros((num_rows, num_cols))

    newarray = np.zeros((num_rows,num_cols))
    # map onto new array the class value
    for i in range(num_rows):
        for j in range(num_cols):
            split_string_array = str(rgb_mask_image[i, j,:3]).split()
            if split_string_array[0] == "[": # if there is a space between the "[" and first int
                newarray[i][j] = mapping_dict["[" + ",".join(str(rgb_mask_image[i, j,:3]).split()[1:])]
            if split_string_array[0] != "[":
                newarray[i][j] = mapping_dict[",".join(split_string_array)]
    newarray = newarray.astype(np.uint8)

    output_image = Image.fromarray(newarray)
    output_image.save(os.path.join("./int_mask/masks/",rgb_mask))