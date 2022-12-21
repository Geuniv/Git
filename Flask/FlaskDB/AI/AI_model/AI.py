import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import cv2


def solution2(filename):
    classes = ['normal_eye' , 'Cataract']
    model = MyModel()
    model.load_state_dict(torch.load('model_dict.pth'), strict=False)
    transform = create_transform()
    image = cv2.imread('.' + filename, cv2.IMREAD_COLOR)
    image = np.array(image)
    image = transform(image=image)['image']
    image = image.unsqueeze(0)
    output = model(image)
    output = torch.softmax(output.detach(), dim=1)
    index = torch.argmax(output)
    label = classes[index]
    #return redirect("label : {}, prob : {}".format(label, output[index]))
    #return render_template('index.html', label = label, probability = output[index])
    return label, str(output[0][index].item())[:5]