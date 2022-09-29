# This code has been adapted from: Visualizing Feature Maps using PyTorch, by Ravi vaishnav, 28/06/2021
# https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573

# WEIGHT VISUALISATION - FEATURE MAPS - MTL MODEL WITHOUT RECONSTRUCTION AUXILIARY TASK

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import load_checkpoint
from models.unet_mtl_class import UNET_MTL, DoubleConv
from loader import PetDataset, PetDatasetGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = UNET_MTL(in_ch=3, out_ch=1).double().to(device)

# Load trained model ... 
###     WRITE LOADING CODE HERE     ### 
load_checkpoint(model,None,'E:/comp0090-cw2/models_weight/mtl_class_diff_20.pt')
print(model)

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())


#counter to keep count of the conv layers
counter = 0

#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == DoubleConv:
        grand_children = list(list(model_children[i].children())[0].children())
        for j in range(len(grand_children)):
            if type(grand_children[j]) == nn.Conv2d and counter < 10:
                counter += 1
                model_weights.append(grand_children[j].weight)
                conv_layers.append(grand_children[j])


print(f"Total convolution layers: {counter}")

for i in range(len(conv_layers)):
    print(f'{i+1}: {conv_layers[i]}')

# Load image we will test on to visualise the filters
testset = PetDataset('datasets-oxpet/test/')
test_generator = PetDatasetGenerator(testset, batch_size=5, shuffle=False)
img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

its = 0
for i, data in enumerate(test_generator):
    images, _, _, _ = data
    images = images.to(device)
    images = img_transform(images)
    its += 1
    if its == 1:
        break

im = images[0] / 255
im_rearr = torch.transpose(torch.transpose(im, 0, 1), 1, 2)
# plt.imshow(im_rearr)
# plt.show()


# Add batch size to the image
im = im.unsqueeze(0)

# Expected object of scalar type Float
# im = im.float()


# Generate feature maps
outputs = []
names = []
for layer in conv_layers[0:]:
    im = layer(im)
    outputs.append(im)
    names.append(str(layer))
print(len(outputs))

#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)


# Now convert 3D tensor to 2D and sum the same element of every channel
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0] # we just average over the depth and end up in 2D
    processed.append(gray_scale.data.cpu().numpy())

for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0] + '-' + str(i+1), fontsize=30)
plt.savefig(str('feature_maps_mtl_class_diff.jpg'), bbox_inches='tight')

