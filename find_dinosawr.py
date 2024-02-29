import torchvision
import torchvision.transforms as transforms
import torch
import cv2

PATH = 'data-samples/'

layer4_features = None

def get_features(module, inputs, output):
    global layer4_features   
    layer4_features = output



model = torchvision.models.resnet50(pretrained=True)
model.eval()
model.layer4.register_forward_hook(get_features)

dinosaur = cv2.imread(PATH + 'dino.jpg')
d_map = cv2.imread(PATH + 'abc.jpg')


transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


import numpy as np

def get_heat_map(dino_map, dino):
    dino_m = cv2.cvtColor(dino_map, cv2.COLOR_BGR2RGB) 
    dino_map_tensor = transform(dino_m).unsqueeze(0)
    model(dino_map_tensor)
    map_features =  layer4_features[:]
    map_features = torch.nn.functional.normalize(map_features, dim=0)
    print(map_features.shape)

    dino_d = cv2.cvtColor(dino, cv2.COLOR_BGR2RGB) 
    dino_tensor = transform(dino_d).unsqueeze(0) 
    model(dino_tensor)
    dino_features = layer4_features[:]
    dino_features = torch.nn.functional.normalize(dino_features, dim=0)
    print(dino_features.shape)


    
    heat_map = torch.conv2d(map_features, dino_features, padding=1)
    heat_map = heat_map.detach().numpy().squeeze()
    # print(heat_map.shape)
    
    heat_map = heat_map / np.max(heat_map) * 255
    heat_map = heat_map.astype(np.uint8)
    heat_map = cv2.resize(heat_map, dino_map.shape[:2])

    # print(dino_map.shape[:2])
    # print(heat_map.shape)
    
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    dino_map = cv2.resize(dino_map, dino_map.shape[:2])
    print(dino_map.shape[:2])
    return cv2.addWeighted(heat_map, 0.7, dino_map, 0.3, 0)


heat_map = get_heat_map(d_map, dinosaur)
print(d_map.shape)
cv2.imshow('dino', dinosaur)
cv2.imshow('heat_map', heat_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
