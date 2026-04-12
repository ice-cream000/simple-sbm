import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

from model import UNet
from utils import fix_seeds, check_folder_and_create

from torchvision import transforms


def _labels(labels, num_images, device):
    if labels is None:
        labels = [0] * num_images
    elif type(labels) == torch.Tensor:
        return labels.to(torch.long).to(device)
    elif type(labels) == int or type(labels) == float or type(labels) == str:
        labels = [int(labels)] * num_images
    elif type(labels) == list:
        if len(labels) != num_images:
            labels = [labels[0]] * num_images
    else:
        print(f'Error: {type(labels)}')
        return None
    return torch.tensor(labels, dtype=torch.long).to(device)


def sampling(model, num_images, min_sigma, max_sigma, len_sigma, K, alpha, gamma, 
             labels, device, euler=True, save_per=0.2, verbose=True):
    r =(min_sigma / max_sigma) ** (1 / (len_sigma - 1))
    sigma_level = torch.tensor([max_sigma * (r ** i) for i in range(len_sigma)]).to(device)

    history = []
    model.eval()
    
    x = torch.randn(num_images, 3, 32, 32).to(device) * sigma_level[0]

    if save_per < 1:
        img_snapshot = x.clone().cpu()
        img_snapshot = (img_snapshot + 1) / 2
        img_snapshot = img_snapshot.clamp(0, 1)
        history.append((0, sigma_level[0], img_snapshot))
    save_interval = max(1, int(len_sigma * save_per))
    
    with torch.no_grad():
        for t in range(len_sigma):
            sigma = sigma_level[t].view(1, 1, 1, 1).repeat(num_images, 1, 1, 1)
            
            alpha_t = alpha * (sigma**2 / sigma_level[-1]**2)

            # 探索
            for k in range(K):
                u_k = torch.randn_like(x)
                
                if euler and t >= len_sigma * 0.9:
                  u_k = torch.zeros_like(x)
                
                # 条件付きスコア
                score_cond = model(x, sigma, labels)
                # 無条件スコア
                score_uncond = model(x, sigma, None)
                # スコア
                score = gamma * score_cond + (1 - gamma) * score_uncond
                x = x + alpha_t * score + torch.sqrt(2 * alpha_t) * u_k

            # save_perごとに生成過程を保存

            if save_per==1:
                should_save = (t == len_sigma - 1)
            else:
                should_save = ((t + 1) % save_interval == 0) or (t == len_sigma - 1)

            if should_save:
                if verbose:
                    print(f'Sigma Step: {t+1}/{len_sigma}, K Step: {k+1}/{K}: Image Saved')
                img_snapshot = x.clone().cpu()
                img_snapshot = (img_snapshot + 1) / 2
                img_snapshot = img_snapshot.clamp(0, 1)
                history.append((t+1, sigma_level[t], img_snapshot))

    return history


def main(args):
    fix_seeds(args.seed) 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    if args.labels_all:
        _ = args.num_images // 10
        num_images = _*10
        labels = torch.tensor(list(range(10)) * _, dtype=torch.long).to(device)
    else:
        num_images = args.num_images
        labels = _labels(args.labels, num_images, device) 

    score_net = UNet(in_channels=3).to(device)
    
    if os.path.exists(args.model_path):
        print(f'Loading model: {args.model_path}')
        state_dict = torch.load(args.model_path, map_location=device)['model_state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
        score_net.load_state_dict(state_dict=state_dict)
    else:
        print(f'Error: Model file not found: {args.model_path}')
        return

    print('='*60)
    print(f'  {"seed":<12}: {args.seed}')
    print(f'  {"num_images":<12}: {num_images}')
    print(f'  {"min_sigma":<12}: {args.min_sigma}')
    print(f'  {"max_sigma":<12}: {args.max_sigma}')
    print(f'  {"len_sigma":<12}: {args.len_sigma}')
    print(f'  {"K":<12}: {args.K}')
    print(f'  {"alpha":<12}: {args.alpha}')
    print(f'  {"gamma":<12}: {args.gamma}')
    print(f'  {"euler":<12}: {args.euler}')
    print('='*60)

    history = sampling(score_net, 
                       num_images=num_images,
                       min_sigma=args.min_sigma, 
                       max_sigma=args.max_sigma, 
                       len_sigma=args.len_sigma, 
                       K=args.K, 
                       alpha=args.alpha, 
                       gamma=args.gamma,
                       labels=labels, 
                       device=device, 
                       euler=args.euler, 
                       save_per=args.save_per, 
                       verbose=args.verbose)

    num_snaps = len(history)
    grid_nrow = int(np.sqrt(args.num_images))
    
    _, axes = plt.subplots(1, num_snaps, figsize=(3.0 * num_snaps, 3.5))
    if num_snaps == 1:
        axes = [axes]
    
    process_title = f'seed={args.seed}_K={args.K}_alpha={args.alpha}_num={args.num_images}_euler={args.euler}'
    
    for i, (t_step, sigma_val, batch_tensor) in enumerate(history):
        ax = axes[i]
        
        grid_img = torchvision.utils.make_grid(batch_tensor, nrow=grid_nrow, padding=2, normalize=False)
        
        img_np = grid_img.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        sigma_val = sigma_val.mean().item()
        if args.show_title:
            ax.set_title(f'$\sigma_{{{t_step}}}$={sigma_val:.3f}', fontsize=32)
        ax.axis('off') 

    plt.tight_layout()

    if args.save_dir:
        check_folder_and_create(args.save_dir)
        if args.save_individual:
            # 生成結果のみを個別保存
            os.makedirs(args.save_dir, exist_ok=True)
            for i in range(args.num_images):
                img = history[-1][2][i].cpu()
                img = img.clamp(0, 1)
                img = transforms.ToPILImage()(img).convert('RGB')
                file_path = os.path.join(args.save_dir, f'label{labels[i].item()}_{args.seed}_{i}.png')
                img.save(file_path)
                print(f'Saved Image: {file_path}')
        else:
            # 生成過程全体を保存
            file_path = os.path.join(args.save_dir, process_title + '.png')
            plt.savefig(file_path, bbox_inches='tight', dpi=150)
            print(f'Saved Image: {file_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score-Based Model Generation')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--len_sigma', type=int, default=250)
    parser.add_argument('--labels', type=int, nargs='+', default=[0])
    parser.add_argument('--min_sigma', type=float, default=0.01)
    parser.add_argument('--max_sigma', type=float, default=50.0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.00002)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--euler', dest='euler', action='store_true')
    parser.add_argument('--labels_all', action='store_true')
    parser.add_argument('--save-individual', dest='save_individual', action='store_true')
    parser.add_argument('--save-per', type=float, default=0.2)
    parser.add_argument('--show-title', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)