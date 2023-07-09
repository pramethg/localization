import os
import torch
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action = 'store_true')
    parser.add_argument('--seed', type = int, default = 9)
    parser.add_argument('--data_aug', action = 'store_true')
    parser.add_argument('--save_json', action = 'store_true')
    parser.add_argument('--pretrained', action = 'store_true')
    parser.add_argument('--save_model', action = 'store_true')
    parser.add_argument('--epochs', type = int, default = 250)
    parser.add_argument('--beta', type = float, default = 0.5)
    parser.add_argument('--lrate', type = float, default = 3e-3)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--save_multiple', action = 'store_true')
    parser.add_argument('--save_dir', type = str, default = "./results/")
    parser.add_argument('--model_file', type = str, default = "/")
    return parser

def save_checkpoint(model, filename = 'model.pth', save_dir = "./baseline"):
    print('Saving Checkpoint')
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
