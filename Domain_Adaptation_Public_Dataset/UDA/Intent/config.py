import torch

# Paths

"""
DATASET1 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids_Test/TN5000"
DATASET2 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids_Test/AUITD"
DATASET3 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids_Test/DDTI"

"""
DATASET2 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids/Dataset_npy/TN5000"
DATASET1 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids/Dataset_npy/AUITD"
DATASET3 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids/Dataset_npy/DDTI"




LOG_FOLDER = "logs"

SEED = 42
TARGET_SHAPE = (224,224)
NUM_EPOCHS = 50
LR = 1e-5
K_FOLDS = 5
BATCH_SIZE=64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_TRAINED=False
SCHEDULER=True
PATIENCE= 10

OPTIMIZER = "AdamW"
WEIGHT_DECAY=1e-5
SAVE=True
ARCHITECTURE ="densenet201"

GRADUAL_FREEZING = 0