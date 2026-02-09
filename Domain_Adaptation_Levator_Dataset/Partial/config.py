import torch

# Paths
"""
DATASET1 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids_Test/TN5000"
DATASET2 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids_Test/AUITD"
DATASET3 = r"/data/leuven/379/vsc37951/Datasets/Dataset_Tyroids_Test/DDTI"
"""

#DATASET1 =  r"/data/leuven/379/vsc37951/Datasets/Final/Adela_imm_Slices"
DATASET1 =  r"/data/leuven/379/vsc37951/Datasets/Final/Adela_Slices"
DATASET2 =  r"/data/leuven/379/vsc37951/Datasets/Final/Sydney_Slices_After"


MASK_FOLDER = r"/data/leuven/379/vsc37951/Datasets/Final/Sydney_Slices_Masks"


LABEL_COLUMN = "Avulsion"
LOG_FOLDER = "logs"


# Configs
SEED = 42
TARGET_SHAPE = (224,224)
NUM_EPOCHS = 100
LR = 1e-5
K_FOLDS = 5

BATCH_SIZE=16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_TRAINED=True
SCHEDULER=True
PATIENCE= 15


OPTIMIZER = "AdamW"
WEIGHT_DECAY=5e-1
SAVE=True

ARCHITECTURE = "vgg16_bn"
ARCHITECTURE = "convnextv2_atto"
ARCHITECTURE = "densenet201"


GRADUAL_FREEZING = 0
