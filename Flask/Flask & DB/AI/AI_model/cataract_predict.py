def image_test(img):
    from PIL import Image
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rc('font', family='NanumGothic') # For Windows
    import warnings
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def imshow(input, title):
        input = input.numpy().transpose((1, 2, 0))
        # 이미지 정규화 해제하기
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
        plt.imshow(input)
        plt.title(title)
        plt.show()

    model = models.resnet34(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)

    model.load_state_dict(torch.load("C:/Users/user/Desktop/pythonProject/pythonProject/Flask & DB/AI/AI_model/model_dict.pth" , map_location=torch.device('cpu')))
    model.eval()

    model = model.to(device)
    image = Image.open(img).convert('RGB')
    image = transforms_test(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        if torch.max(outputs) == list(outputs[0])[0]:
            class_name = 'Cataract'
        elif torch.max(outputs) == list(outputs[0])[1]:
            class_name = 'Normal'
        print(class_name)
        imshow(image.cpu().data[0], title='Predict: ' + class_name)