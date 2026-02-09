from config import *  
from models import BaseModel
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import warnings
import logging
import os
from datetime import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit,StratifiedShuffleSplit
from collections import Counter
from sklearn.model_selection import train_test_split

def save_images(loader, save_folder="saved_images", max_images=None):
    
    output_dir = os.path.join(os.getcwd(), save_folder)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for imgs, labels, fnames in loader:
        for img, fname in zip(imgs, fnames):
            if max_images is not None and count >= max_images:
                break

            image_np = img.numpy()
            if image_np.shape[0] in [1, 3]:
                image_np = np.transpose(image_np, (1,2,0))
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

            save_path = os.path.join(output_dir, f"{fname}.png")
            plt.imsave(save_path, image_np)
            count += 1
        if max_images is not None and count >= max_images:
            break
    print(f"All images saved")


def training_config(arc=ARCHITECTURE, device=DEVICE, optimizer=OPTIMIZER,weight_decay=WEIGHT_DECAY,pre_trained=PRE_TRAINED):
    global params

    model = BaseModel(backbone_name=arc,in_channels=3,num_classes=1,device=device,pretrained=False)    
    model = model.to(device) 
    params = (sum(p.numel() for p in model.parameters())) / 1e6

    if OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)

    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {OPTIMIZER} not supported.")

    scheduler = None
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    """
    pos_weight = torch.tensor([2.0]).to(device)  # peso para a classe positiva
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    """
    criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, scheduler, criterion, params

def set_seed(seed=42):
    import os
    import random
    import numpy as np
    import torch
    import monai

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # Importante para CUDA determinism (PyTorch >= 1.8)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # MONAI usa seu próprio gerador global
    #monai.utils.set_determinism(seed=seed)

    print(f"Seed fixada em {seed}")

    

def global_configs(ARCHITECTURE):
    global FOLDER, DEVICE, PRE_TRAINED, params,TARGET_SHAPE, LR, K_FOLDS, BATCH_SIZE, SCHEDULER, OPTIMIZER, WEIGHT_DECAY, LOG_FOLDER

    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)

    filename = f"{ARCHITECTURE}_Pre{PRE_TRAINED}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    log_path = os.path.join(LOG_FOLDER, filename)

    file_handler = logging.FileHandler(log_path)
    logging.getLogger().addHandler(file_handler)

    logging.info("")
    logging.info("================================== Starting of Training ==================================")
    logging.info(f"Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Starting Time: {datetime.now()}")
    logging.info(f"Architecture Selected: {ARCHITECTURE} | Number of Parameters: {params:.2f} M | Pre_Trained: {PRE_TRAINED} ")
    logging.info(f"Target Shape: {TARGET_SHAPE} | Learning Rate: {LR} | K_Folds: {K_FOLDS} | Batch Size: {BATCH_SIZE}")
    logging.info(f"Scheduler: {SCHEDULER} | Optimizer: {OPTIMIZER} | Weight_Decay: {WEIGHT_DECAY}")

def print_overlay (df,train_idx_final,val_idx_final,test_idx_final):
    train_set_idx = set(train_idx_final)
    val_set_idx = set(val_idx_final)
    test_set_idx = set(test_idx_final)

    logging.info(f"Overlap: {any([train_set_idx & val_set_idx, train_set_idx & test_set_idx, val_set_idx & test_set_idx])}") 
        
    train_final_df = df.loc[train_idx_final]
    train_counts = train_final_df['dataset'].value_counts()
    logging.info("Train set dataset counts:")
    for dataset, count in train_counts.items():
        logging.info(f"{dataset}: {count}")

    val_final_df = df.loc[val_idx_final]
    val_counts = val_final_df['dataset'].value_counts()
    logging.info("Validation set dataset counts:")
    for dataset, count in val_counts.items():
        logging.info(f"{dataset}: {count}")

    test_final_df = df.loc[test_idx_final]
    test_counts = test_final_df['dataset'].value_counts()
    logging.info("Test set dataset counts:")
    for dataset, count in test_counts.items():
        logging.info(f"{dataset}: {count}")

def print_loader_distribution(dataset, FOLD, train_loader, val_loader, test_loader):
    label_counts = Counter()
    labelval_counts = Counter()
    labeltest_counts = Counter()

    # Train loader
    for i, item in enumerate(train_loader):
        labels = item['label']
        label_counts.update(labels.tolist())

    # Validation loader
    for i, item in enumerate(val_loader):
        labels = item['label']
        labelval_counts.update(labels.tolist())

    # Test loader
    for i, item in enumerate(test_loader):
        labels = item['label']
        labeltest_counts.update(labels.tolist())

    logging.info(f"\n{'='*90} FOLD {FOLD} {'='*90}\n")
    logging.info(f"Train label distribution: {label_counts}")
    logging.info(f"Val label distribution:   {labelval_counts}")
    logging.info(f"Test label distribution:  {labeltest_counts}\n")


def save_results_fold (fold_results,DATASET,FOLD,best_train_metrics,best_val_metrics,test_metrics):

    fold_results.append({
        "dataset": DATASET,      
        "fold": FOLD,            
        "train_loss": best_train_metrics[0],
        "train_acc": best_train_metrics[1],
        "train_sens": best_train_metrics[2],
        "train_spec": best_train_metrics[3],
        "train_f1": best_train_metrics[4],
        "val_loss": best_val_metrics[0],
        "val_acc": best_val_metrics[1],
        "val_sens": best_val_metrics[2],
        "val_spec": best_val_metrics[3],
        "val_f1": best_val_metrics[4],
        "val_auc": best_val_metrics[5],
        "test_loss": test_metrics[0],
        "test_acc": test_metrics[1],
        "test_sens": test_metrics[2],
        "test_spec": test_metrics[3],
        "test_f1": test_metrics[4],
        "test_auc": test_metrics[5]
    })


def print_tensorboard(writer,FOLD, DATASET, train_loss,train_f1, val_loss, val_sens,val_spec,val_f1,val_auc,epoch):

    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/Loss", train_loss, epoch)
    #writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/Loss_Class", train_loss_class, epoch)
    #writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/Loss_MMD", train_loss_mmd, epoch)    
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/F1", train_f1, epoch)

    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Loss", val_loss, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Sens", val_sens, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Spec", val_spec, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/F1", val_f1, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/AUC", val_auc, epoch)




def print_results_fold(FOLD,fold_results):
    logging.info(f"\n================================= Best Results for FOLD {FOLD} ==================================")
    logging.info(
        f"Train - Loss: {fold_results[-1]['train_loss']:.4f} | "
        f"Acc: {fold_results[-1]['train_acc']:.4f} | "
        f"Sens: {fold_results[-1]['train_sens']:.4f} | "
        f"Spec: {fold_results[-1]['train_spec']:.4f} | "
        f"F1: {fold_results[-1]['train_f1']:.4f}"
    )
    logging.info(
        f"Val   - Loss: {fold_results[-1]['val_loss']:.4f} | "
        f"Acc: {fold_results[-1]['val_acc']:.4f} | "
        f"Sens: {fold_results[-1]['val_sens']:.4f} | "
        f"Spec: {fold_results[-1]['val_spec']:.4f} | "
        f"F1: {fold_results[-1]['val_f1']:.4f} | "
        f"AUC: {fold_results[-1]['val_auc']:.4f}"
    )
    logging.info(
        f"Test  - Loss: {fold_results[-1]['test_loss']:.4f} | "
        f"Acc: {fold_results[-1]['test_acc']:.4f} | "
        f"Sens: {fold_results[-1]['test_sens']:.4f} | "
        f"Spec: {fold_results[-1]['test_spec']:.4f} | "
        f"F1: {fold_results[-1]['test_f1']:.4f} | "
        f"AUC: {fold_results[-1]['test_auc']:.4f}"
    )
    logging.info("============================================================================================\n")



def print_final_metrics(fold_results):

    import pandas as pd
    df = pd.DataFrame(fold_results)

    logging.info("\n====================== K-Fold Summary ======================")

    for dataset_name, group in df.groupby("dataset"):
        logging.info(f"\n--- Dataset: {dataset_name} ---")
        
        # ------------------------- TRAIN -------------------------
        logging.info("\n===== Training Metrics =====")
        logging.info(f"Loss:         {group['train_loss'].mean():.4f} ± {group['train_loss'].std():.4f}")
        logging.info(f"Accuracy:     {group['train_acc'].mean():.4f} ± {group['train_acc'].std():.4f}")
        logging.info(f"Sensitivity:  {group['train_sens'].mean():.4f} ± {group['train_sens'].std():.4f}")
        logging.info(f"Specificity:  {group['train_spec'].mean():.4f} ± {group['train_spec'].std():.4f}")
        logging.info(f"F1-score:     {group['train_f1'].mean():.4f} ± {group['train_f1'].std():.4f}")

        # ----------------------- VALIDATION -----------------------
        logging.info("\n===== Validation Metrics =====")
        logging.info(f"Loss:         {group['val_loss'].mean():.4f} ± {group['val_loss'].std():.4f}")
        logging.info(f"Accuracy:     {group['val_acc'].mean():.4f} ± {group['val_acc'].std():.4f}")
        logging.info(f"Sensitivity:  {group['val_sens'].mean():.4f} ± {group['val_sens'].std():.4f}")
        logging.info(f"Specificity:  {group['val_spec'].mean():.4f} ± {group['val_spec'].std():.4f}")
        logging.info(f"F1-score:     {group['val_f1'].mean():.4f} ± {group['val_f1'].std():.4f}")
        logging.info(f"AUC:          {group['val_auc'].mean():.4f} ± {group['val_auc'].std():.4f}")

        # ------------------------- TEST --------------------------
        logging.info("\n===== Test Metrics =====")
        logging.info(f"Loss:         {group['test_loss'].mean():.4f} ± {group['test_loss'].std():.4f}")
        logging.info(f"Accuracy:     {group['test_acc'].mean():.4f} ± {group['test_acc'].std():.4f}")
        logging.info(f"Sensitivity:  {group['test_sens'].mean():.4f} ± {group['test_sens'].std():.4f}")
        logging.info(f"Specificity:  {group['test_spec'].mean():.4f} ± {group['test_spec'].std():.4f}")
        logging.info(f"F1-score:     {group['test_f1'].mean():.4f} ± {group['test_f1'].std():.4f}")
        logging.info(f"AUC:          {group['test_auc'].mean():.4f} ± {group['test_auc'].std():.4f}")

    logging.info("\n============================================================\n")

def create_split(df):

    folds_dict = {}
    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name]
        x = subset.index.values
        y = subset['label'].values
        groups = subset["id"].values 

        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(x, y,groups)):
            test_set = subset.iloc[test_idx]

            train_val_set = subset.iloc[train_val_idx]

            gss = GroupShuffleSplit(test_size=0.25, random_state=42)
            train_idx, val_idx = next(gss.split(train_val_set, train_val_set['label'], groups=train_val_set['id']))
            train_set = train_val_set.iloc[train_idx]
            val_set = train_val_set.iloc[val_idx]
            folds.append({'train': train_set,'val': val_set,'test': test_set}) 

        folds_dict[dataset_name] = folds
    return folds_dict

def stratified_target_split(target_df, label_ratios , n_samples, random_state=42):
    # Calcular proporção de cada label no source

    sampled_indices = []
    rng = np.random.default_rng(seed=random_state)

    for label, ratio in label_ratios.items():
        n_label_samples = int(np.round(n_samples * ratio))
        label_indices = target_df[target_df['label'] == label].index.tolist()
        if len(label_indices) <= n_label_samples:
            sampled = label_indices
        else:
            sampled = rng.choice(label_indices, size=n_label_samples, replace=False).tolist()
        sampled_indices.extend(sampled)
    
    return target_df.loc[sampled_indices]

def stratified_target_split_exact(target_df, n_samples_per_label, random_state=42):
    
    sampled_indices = []
    rng = np.random.default_rng(seed=random_state)

    for label, n_label_samples in n_samples_per_label.items():
        # Pegar índices do target para essa label
        label_indices = target_df[target_df['label'] == label].index.tolist()
        
        # Se o target tem menos exemplos que o desejado, pega todos
        if len(label_indices) <= n_label_samples:
            sampled = label_indices
        else:
            sampled = rng.choice(label_indices, size=n_label_samples, replace=False).tolist()
        
        sampled_indices.extend(sampled)
    
    return target_df.loc[sampled_indices]

def create_set(df,PERCENTAGE,folds_dict,DATASET,FOLD):

    train_dict = {}
    val_dict = {}

    train_idx = folds_dict[DATASET][FOLD]['train'].index.tolist()  
    val_idx = folds_dict[DATASET][FOLD]['val'].index.tolist()  
    test_idx_final = folds_dict[DATASET][FOLD]['test'].index.tolist()  
    train_idx_final,val_idx_final = [],[]

    other_train_idx=[]
    other_val_idx=[]

    train_val_indices = []
    for other_dataset in df['dataset'].unique():
        if other_dataset == DATASET:
            continue  
        train = folds_dict[other_dataset][FOLD]['train'].index.tolist()
        other_train_idx.extend(train)
        val = folds_dict[other_dataset][FOLD]['val'].index.tolist()
        other_val_idx.extend(val)

        train_dict[other_dataset] = train
        val_dict[other_dataset] = val
    

    train_counts = {dataset: len(indices) for dataset, indices in train_dict.items()}
    val_counts = {dataset: len(indices) for dataset, indices in val_dict.items()}

    num_train = list(train_counts.values())[0]
    num_val = list(val_counts.values())[0]
    
    train_target = df.loc[train_idx]
    val_target = df.loc[val_idx]

    first_train_dataset = list(train_dict.keys())[0]
    train_idx_source = train_dict[first_train_dataset]
    first_val_dataset = list(val_dict.keys())[0]
    val_idx_source = val_dict[first_val_dataset] 

    train_source = df.loc[train_idx_source]
    val_source = df.loc[val_idx_source]

 
    if len(train_idx) > num_train:

        # TRAIN
        n_samples_train = PERCENTAGE * num_train
        expected_numbers_train = train_source['label'].value_counts()

        if n_samples_train != 0:
            train_ratios = train_source['label'].value_counts(normalize=True)
            train_df_reduced = stratified_target_split(train_target, train_ratios, n_samples_train)
            train_idx_final = train_df_reduced.index.tolist()
            current_numbers_train = train_df_reduced['label'].value_counts()
            source_numbers_train = expected_numbers_train - current_numbers_train
           
        else:
            source_numbers_train = expected_numbers_train
        source_df_reduced_train = stratified_target_split_exact(train_source, source_numbers_train)
        source_idx_final_train = source_df_reduced_train.index.tolist()
        train_idx_final.extend(source_idx_final_train)


        # VAL
        n_samples_val = PERCENTAGE * num_val
        expected_numbers_val = val_source['label'].value_counts()

        if n_samples_train != 0:
            val_ratios = val_source['label'].value_counts(normalize=True)
            val_df_reduced   = stratified_target_split(val_target, val_ratios, n_samples_val)
            val_idx_final   = val_df_reduced.index.tolist()
            current_numbers_val = val_df_reduced['label'].value_counts()
            source_numbers_val = expected_numbers_val - current_numbers_val
            
        else:
            source_numbers_val = expected_numbers_val 

        source_df_reduced_val = stratified_target_split_exact(val_source, source_numbers_val)
        source_idx_final_val = source_df_reduced_val.index.tolist()
        val_idx_final.extend(source_idx_final_val)


    else: 
        
        # TRAIN
        n_samples_train = round(PERCENTAGE * len(train_idx))
        expected_numbers_train = train_target['label'].value_counts()

        if n_samples_train !=0 :
            train_ratios = train_target['label'].value_counts(normalize=True)
            train_df_reduced = stratified_target_split(train_target, train_ratios, n_samples_train)
            train_idx_final = train_df_reduced.index.tolist()
            current_numbers_train = train_df_reduced['label'].value_counts()
            source_numbers_train = expected_numbers_train - current_numbers_train

        else:
            source_numbers_train = expected_numbers_train 

        source_df_reduced_train = stratified_target_split_exact(train_source, source_numbers_train)
        source_idx_final_train = source_df_reduced_train.index.tolist()
        train_idx_final.extend(source_idx_final_train)
        

        # VAL
        n_samples_val = round(PERCENTAGE * len(val_idx))
        expected_numbers_val = val_target['label'].value_counts()

        if n_samples_val != 0:
            val_ratios = val_target['label'].value_counts(normalize=True)
            val_df_reduced   = stratified_target_split(val_target, val_ratios, n_samples_val)
            val_idx_final   = val_df_reduced.index.tolist()
            current_numbers_val = val_df_reduced['label'].value_counts()
            source_numbers_val = expected_numbers_val - current_numbers_val
        else:
            source_numbers_val = expected_numbers_val

        source_df_reduced_val = stratified_target_split_exact(val_source, source_numbers_val)
        source_idx_final_val = source_df_reduced_val.index.tolist()
        val_idx_final.extend(source_idx_final_val)

     
    return train_idx_final,val_idx_final,test_idx_final