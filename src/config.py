import torch

class CFG:
    # Paths (used in Kaggle)
    CSV_PATH = "/kaggle/input/data/Data_Entry_2017.csv"
    IMAGE_DIR = "/kaggle/input/data/images_006/images"

    # Training parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-4
    SUBSET_SIZE = 2000  # set to None for full data

    # Labels
    LABELS = [
        "Atelectasis","Cardiomegaly","Effusion","Infiltration",
        "Mass","Nodule","Pneumonia","Pneumothorax",
        "Consolidation","Edema","Emphysema","Fibrosis",
        "Pleural_Thickening","Hernia"
    ]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
