import torch
from config import *  
from models import MMDModel
from datasets import Dataset2D
from train_func import train_one_epoch, validate, train_kfold
from func import set_seed, training_config, global_configs, print_final_metrics
import time
import logging

def main():

    DATASET_MANY= [DATASET1,DATASET2]
    start_time = time.time()
    set_seed()
    _, optimizer, scheduler, criterion, params = training_config()
    global_configs(ARCHITECTURE)
    dataset = Dataset2D(DATASET_MANY, target_shape=(224, 224))

    fold_results = train_kfold(dataset,DATASET_MANY,ARCHITECTURE, criterion, optimizer, scheduler,NUM_EPOCHS, BATCH_SIZE, DEVICE, patience=PATIENCE)
    print_final_metrics(fold_results)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Running for: {elapsed_time/60:.2f} min ({elapsed_time:.2f} s)")

if __name__ == "__main__":
    main()
