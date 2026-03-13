from config.intel_img_cfg import Itel_Data_Config, Model_Config
from src.dataset import data_processing
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
import yaml
import mlflow
import torch

def train():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    lr = params["lr"]
    momentum = params["momentum"]
    epochs = params["epochs"]

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(Model_Config.DEVICE)
    model.fc = nn.Linear(model.fc.in_features, Itel_Data_Config.N_CLASSES)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    train_loader, test_loader = data_processing()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    mlflow.start_run()
    mlflow.log_params(params)

    for epoch in range(epochs):
        model.train()
        process_bar = tqdm(train_loader, colour='cyan')
        train_loss = []

        for iter, (images, labels) in enumerate(process_bar):
            images = images.to(Model_Config.DEVICE)
            labels = labels.to(Model_Config.DEVICE)

            outputs = model(images)
            loss_value = criterion(outputs, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loss.append(loss_value.item())
            process_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, epochs, loss_value))


        mean_loss = np.mean(train_loss)
        mlflow.log_metric("loss", mean_loss, step=epoch)

        model.eval()
        process_bar = tqdm(test_loader, colour='yellow')
        test_loss = []
        for iter, (images, labels) in enumerate(process_bar):
            images = images.to(Model_Config.DEVICE)
            labels = labels.to(Model_Config.DEVICE)
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            test_loss.append(loss_value.item())

            process_bar.set_description("Image {}/{}".format(len(test_loader), iter))

        mean_test_loss = np.mean(test_loss)
        mlflow.log_metric("test_loss", mean_test_loss, step=epoch)

    torch.save(model.state_dict(), "models/resnet50.pth")
    mlflow.log_artifact("models/resnet50.pth")

    mlflow.end_run()

if __name__ == '__main__':
    train()