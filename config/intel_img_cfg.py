from pathlib import Path

class Itel_Data_Config():
    ROOT_DIR = Path(__file__).parent.parent
    ROOT_DATA_TRAIN = ROOT_DIR / 'data' / 'seg_train'
    ROOT_DATA_TEST = ROOT_DIR / 'data' / 'seg_test'
    ROOT_DATA_PRED = ROOT_DIR / 'data' / 'seg_pred'

    N_CLASSES = 6
    IMG_SIZE = 64
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD =[0.229, 0.224, 0.225]

class Model_Config():
    ROOT_DIR = Path(__file__).parent.parent
    DEVICE = 'cpu'
    MODEL_NAME = 'resnet50'
    MODEL_WEIGHTS = ROOT_DIR / 'models' / 'weights' / 'intel_img_weights.pt'