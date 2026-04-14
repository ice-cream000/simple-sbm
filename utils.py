import os
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime, date

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_folder_and_create(path):
    if not os.path.exists(path): 
        os.makedirs(path)
        print(f'Created New Folder: {path}')
    return path

def save_training_log(path, save_model_name, train_losses, test_losses, loss_config):
    check_folder_and_create(path)
    filepath = os.path.join(path, f'{save_model_name}_loss.json')
    
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d'),
        'config': loss_config,
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f'Saved Loss: {filepath}')

    plt_filepath = os.path.join(path, f'{save_model_name}_loss.png')
    plt.figure(figsize=(10, 6))

    test_epochs = [epoch for epoch, _ in test_losses]
    test_losses_values = [loss for _, loss in test_losses]

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(test_epochs, test_losses_values, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plt_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Loss Plot: {plt_filepath}')