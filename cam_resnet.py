import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import cv2
import json


avgpool_features = None
layer4_features = None

def get_features(module, inputs, output):
    global avgpool_features   
    avgpool_features = output

def get_map(module, inputs, output):
    global layer4_features   
    layer4_features = output

model = torchvision.models.resnet50(pretrained=True)

model.avgpool.register_forward_hook(get_features)
model.layer4.register_forward_hook(get_map)


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

model.eval()

def get_heat_map(img_st):

    img = cv2.cvtColor(img_st, cv2.COLOR_BGR2RGB) 
    img_tensor = transform(img).unsqueeze(0) 

    pred = model(img_tensor)
    pred_ind = torch.argmax(pred)
    weights = model.fc.weight[pred_ind]

    sum = torch.zeros(7, 7)
    for weight, img in zip(weights, layer4_features.squeeze(0)):
        sum += weight * img

    img = sum.detach().numpy()
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img_st, 0.5, heatmap, 0.5, 0)

    return cv2.hconcat([result, img_st]), int(pred_ind)


def get_dict(file_path):
    out_data = dict()

    with open(file_path, 'r') as file:
        for line in file:
            number, string = line.strip().split(':')
            out_data[number] = string[1:]

    return out_data




d = get_dict("imagenet1000.txt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка открытия видеопотока")
else:
    while True:
        ret, frame = cap.read()        
        frame = cv2.resize(frame, (224, 224))

        if not ret:
            print("Не удалось считать кадр")
            break
        
        result, pred_ind = get_heat_map(frame)
        result = cv2.resize(result, (1000, 500))
        image_with_text = cv2.putText(result, d[str(pred_ind)], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Camera", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
