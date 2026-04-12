from utils import fix_seeds, check_folder_and_create
from run import sampling
from model import UNet

import torch
import argparse
import subprocess
import torchvision
from torchvision import transforms
import os
import glob

def main(args):
    if args.num_images % args.batch_size != 0:
        print('Warning: num_images should be divisible by batch_size. The last batch will have fewer images.')
        return
    # リアル画像と生成画像のフォルダをそれぞれ作成
    real_data_path = os.path.join(args.fid_dir, 'real')
    generated_data_path = os.path.join(args.fid_dir, 'generated')
    check_folder_and_create(real_data_path)
    check_folder_and_create(generated_data_path)

    check_real_img = glob.glob(os.path.join(real_data_path, '*.png'))
    if len(check_real_img) == 0:
        # CIFAR10のテストセットをリアル画像として保存
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                 transform=transforms.ToTensor())
        for i, (img, _) in enumerate(dataset):
            img = transforms.ToPILImage()(img).convert('RGB')
            img_path = os.path.join(real_data_path, f'{i}.png')
            img.save(img_path)


    # 生成画像を保存
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
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

    labels = torch.tensor(list(range(10)) * (args.batch_size // 10), dtype=torch.long).to(device)

    max_roop = args.num_images // args.batch_size

    print('='*60)
    print(f'  {"num_images":<12}: {args.num_images}')
    print(f'  {"batch_size":<12}: {args.batch_size}')
    print(f'  {"min_sigma":<12}: {args.min_sigma}')
    print(f'  {"max_sigma":<12}: {args.max_sigma}')
    print(f'  {"len_sigma":<12}: {args.len_sigma}')
    print(f'  {"K":<12}: {args.K}')
    print(f'  {"alpha":<12}: {args.alpha}')
    print(f'  {"gamma":<12}: {args.gamma}')
    print(f'  {"euler":<12}: {args.euler}')
    print(f'  {"seed(default)":<12}: {args.seed}')
    print('='*60)

    for i in range(max_roop): 
        # 既に生成された画像はスキップする
        check_img_path = os.path.join(generated_data_path, f'{i* args.batch_size}.png')
        if os.path.exists(check_img_path):
            print(f'Batch {i+1}/{max_roop}: Already exists, skipping...')
            continue

        fix_seeds(args.seed + i)  
        history = sampling(score_net, 
                           args.batch_size, args.min_sigma, args.max_sigma, args.len_sigma, args.K, args.alpha, args.gamma, 
                           labels, device, args.euler, save_per=1.0, verbose=False)
        imgs = history[-1][2]  
        for j, img_tensor in enumerate(imgs):
            img = img_tensor.clamp(0, 1)
            img = transforms.ToPILImage()(img).convert('RGB')
            img_path = os.path.join(generated_data_path, f'{i * args.batch_size + j}.png')
            img.save(img_path)
        print(f'Batch {i+1}/{max_roop}: Done')
    

    # FIDスコアの計算
    subprocess.run(['python3', '-m', 'pytorch_fid', args.fid_dir + '/real', args.fid_dir + '/generated'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Create images using training model and calculate FID score.'))
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--len_sigma', type=int, default=500)
    parser.add_argument('--labels', type=int, nargs='+', default=[0])
    parser.add_argument('--min_sigma', type=float, default=0.01)
    parser.add_argument('--max_sigma', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--euler', dest='euler', action='store_true')
    parser.add_argument('--fid_dir', type=str, default='./fid_datas')

    args = parser.parse_args()
    main(args)

    
