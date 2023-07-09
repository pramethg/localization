import os
import json
import tqdm
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset

from utils import *
from models import *
from dataset import *
from dsntnn.dsntnn import *

if __name__ == "__main__":
    args = argparser().parse_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = "cpu"

    transforms_tensor = transforms.Compose([transforms.ToTensor()])
    if args.data_aug:
        transforms_horizontal = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5)
                            ])
        transforms_vertical = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomVerticalFlip(0.5)
                            ])
        train_dataset = ConcatDataset(datasets = [
                                LocalizationDataset(root = './data/train/', transforms = transforms_tensor),
                                LocalizationDataset(root = './data/train/', transforms = transforms_vertical),
                                LocalizationDataset(root = './data/train/', transforms = transforms_horizontal),
        ])
    else:
        train_dataset = LocalizationDataset(root = './data/train/', transforms = transforms_tensor)
    test_dataset = LocalizationDataset(root = './data/test/', transforms = transforms_tensor)

    train_loader = DataLoader(dataset = train_dataset,
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers = args.num_workers,
                                pin_memory = True
                            )
    test_loader = DataLoader(dataset = test_dataset,
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers = args.num_workers,
                                pin_memory = True
                            )
    model = LocalizationNet(in_channels = 1, 
                                out_channels = 1,
                                features = [2**i for i in range(4, 9)] + [2**8],
                            )
    optimizer = optim.Adam(model.parameters(), lr = args.lrate)
    
    if args.pretrained:
        print(f'Loading Pre-Trained Model: `{args.save_dir}/{args.model_file}`')
        checkpoint = torch.load(f = f'{args.save_dir}/{args.model_file}')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epochs -= checkpoint['epoch']

    print('Started Training')
    model.train()
    train_loss = {}
    for epoch in range(args.epochs):
        epoch_loss = []
        progressbar = tqdm(train_loader, leave = True)
        for idx, (img, label) in enumerate(progressbar):
            img = Variable(img.to(device = device).float(), requires_grad = True)
            label = Variable(label.to(device = device).float(), requires_grad = True)
            coords, heatmaps, model_out = model(img)
            euclidean_loss = euclidean_losses(label, coords)
            reguralization_loss = js_reg_losses(heatmaps, label, sigma_t = 1.0)
            loss = average_loss(euclidean_losses, reguralization_loss)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.5)
            loss.backward()
            optimizer.step()

            progressbar.update(1)
            progressbar.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
            progressbar.set_postfix(EUC_LOSS = euclidean_loss.item(), REG_LOSS = reguralization_loss.item(), TOTAL_LOSS = loss.item())

        mean_loss = sum(epoch_loss) / len(epoch_loss)
        train_loss[epoch] = float(mean_loss)
        print(f'Epoch [{epoch + 1}/{args.epochs}] TRAIN_LOSS: {mean_loss}')

        if args.save_multiple:
            if (epoch + 1) % 250 == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"EPOCH-{epoch + 1}-" + args.model_file))
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_file))
            print(f'Save the Model: {args.epochs} Epochs')
        
        if args.save_json:
            with open(os.path.join(args.save_dir, 'train_loss.json'), 'w') as f:
                json.dump(train_loss, f)