import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from datetime import date

from model import UNet
from dataset import DatasetCifar10, get_transform
from utils import fix_seeds, check_folder_and_create, save_training_log

from torch.optim import lr_scheduler

def validate(model, dataloader, sigma_level, len_sigma, device):
    model.eval()
    total_loss = torch.tensor([0.0], device=device)
    count = 0

    with torch.no_grad():
        for x in dataloader:
            x, label = x
            x, label = x.to(device), label.to(device).long() 
            if np.random.rand() < 0.1:
                label = None

            current_batch_size = x.shape[0]

            t = torch.randint(0, len_sigma, (current_batch_size,)).to(device)
            sigma = sigma_level[t].view(-1,1,1,1)
            
            # xと同じ形状(batch_size, 1,28,28)の多次元正規分布に従うノイズを生成
            epsilon = torch.randn_like(x)
            noise = sigma * epsilon
            tilde_x = x + noise

            score = model(tilde_x, sigma, label)
            target = -1 * noise / (sigma**2)

            w = sigma**2
            loss = torch.mean(w * (target - score)**2)
            total_loss += loss.detach()
            count += 1

    model.train()
    return (total_loss / count).item()

def main(args):
    fix_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    save_folder = check_folder_and_create(args.save_dir)
    # sigma_level = torch.linspace(args.max_sigma, args.min_sigma, args.len_sigma).to(device)
    r =(args.min_sigma / args.max_sigma) ** (1 / (args.len_sigma - 1))
    sigma_level = torch.tensor([args.max_sigma * (r ** i) for i in range(args.len_sigma)]).to(device)

    score_net = UNet(in_channels=3).to(device)
    # optimizer = optim.Adam(score_net.parameters(), lr=args.lr)
    optimizer = optim.AdamW(score_net.parameters(), lr=args.lr,weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    transform = get_transform()
    train_dataset = DatasetCifar10(path='./data', transform=transform, train=True)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    test_dataset = DatasetCifar10(path='./data', transform=transform, train=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    all_train_batch_num = len(trainloader)
    train_losses = []
    test_losses = []

    loss_config = vars(args)
    loss_config['device'] = device

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        epoch_start = checkpoint['epoch'] + 1
        score_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint['train_loss']
        test_losses = checkpoint['test_loss']
        print(f'Loaded checkpoint from {args.checkpoint}, starting from epoch {epoch_start}')
    else:
        epoch_start = 0 
    score_net = torch.compile(score_net) 
        
    for epoch in range(epoch_start, args.epochs):
        score_net.train()
        epoch_train_loss = torch.tensor([0.0], device=device)
        for i, x in enumerate(trainloader):
            x, label = x
            x, label = x.to(device), label.to(device).long() 

            # バッチ全体でラベルを無効化する
            if np.random.rand() < 0.1:
                label = None

            current_batch_size = x.shape[0]

            t = torch.randint(0, args.len_sigma, (current_batch_size,)).to(device)
            sigma = sigma_level[t].view(-1,1,1,1)

            optimizer.zero_grad(set_to_none=True)

            epsilon = torch.randn_like(x)
            noise = sigma * epsilon
            tilde_x = x + noise

            score = score_net(tilde_x, sigma, label)
            target = - 1 * noise / (sigma ** 2)

            w = sigma**2
            loss = torch.mean(w * (target - score)**2)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], batch [{i+1}/{all_train_batch_num}] Loss: {loss.item():.4f}')
        scheduler.step()

        avg_train_loss = (epoch_train_loss / all_train_batch_num).item()
        train_losses.append(avg_train_loss)

        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            test_loss = validate(score_net, testloader, sigma_level, args.len_sigma, device)
            test_losses.append((epoch+1, test_loss))
            print(f'Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            model_path = os.path.join(save_folder, f'UNet_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': score_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses,
                'test_loss': test_losses
            }, model_path)
            print(f'Saved Model: {model_path}')
        else:
            print(f'Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}')

    save_training_log(save_folder, 'UNet', train_losses, test_losses, loss_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score-Based Model')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--len_sigma', type=int, default=250)
    parser.add_argument('--min_sigma', type=float, default=0.01)
    parser.add_argument('--max_sigma', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--save_dir', type=str, default='Unet')
    
    args = parser.parse_args()
    main(args)