from config.intel_img_cfg import Itel_Data_Config, Model_Config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import yaml

def data_processing():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    batch_size = params["batch_size"]

    transform = Compose([
        Resize((Itel_Data_Config.IMG_SIZE, Itel_Data_Config.IMG_SIZE)),
        ToTensor(),
        Normalize(mean=Itel_Data_Config.NORMALIZE_MEAN,
                  std=Itel_Data_Config.NORMALIZE_STD),
    ])

    train_data = ImageFolder(
        root=Itel_Data_Config.ROOT_DATA_TRAIN,
        transform=transform
    )

    test_data = ImageFolder(
        root=Itel_Data_Config.ROOT_DATA_TEST,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
    )

    return train_loader, test_loader