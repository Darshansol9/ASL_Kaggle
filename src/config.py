DEVICE = 'cuda:0'
DATA_PATH = r'D:\courses\kaggle_dl_proj\input\data'
DIR_PATH = r'D:\courses\kaggle_dl_proj\input'
SRC_PATH = r'D:\courses\kaggle_dl_proj\src'
TEST_PATH = r'D:\courses\kaggle_dl_proj\input\asl_alphabet_test'

FOLDS = {   0:[1,2,3,4],
            1:[0,2,3,4],
            2:[0,1,3,4],
            3:[0,1,2,4],
            4:[0,1,2,3]
        }
NUM_FOLDS = 5
IMG_HEIGHT = 64
IMG_WIDTH = 64
EPOCHS = 10
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1
MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)
BASE_MODEL = 'resnet34'
