from models import centernet
from utils import common
from data import dataset

import torch
import torch.cuda.amp

import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for forward')
    parser.add_argument('--lr', default=0.0008, type=float, help='Learning rate')
    parser.add_argument('--milestone', default=[40], type=list, help='Milestones for learning rate scheduler')
    parser.add_argument('--lam1', default=0.05, type=float)
    parser.add_argument('--lam2', default=0.1, type=float)
    
    parser.add_argument('--img-w', default=512, type=int)
    parser.add_argument('--img-h', default=512, type=int)

    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--total-epoch', type=int, default=50, help='total_epoch')

    parser.add_argument('--source', type=str, default="./data/cityscape.yaml", help='Source dataset from source domain')
    parser.add_argument('--target', type=str, default="./data/cityscape_foggy.yaml", help='Target dataset from target domain')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--save-folder', default='./weights', type=str, help='where you save weights')
    parser.add_argument('--seed', default=7777, type=int)

    opt = parser.parse_args()
    common.setup_seed(opt.seed)
    common.mkdir(dir=opt.save_folder, remove_existing_dir=False)
    
    src_dataset_dict = common.parse_yaml(opt.source)
    tgt_dataset_dict = common.parse_yaml(opt.target)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = centernet.CenterNet(num_classes=len(src_dataset_dict['classes']), pretrained_backbone=True)
    model = model.to(device=device)
    
    src_training_set = dataset.DetectionDataset(root=src_dataset_dict['root'], 
                                                dataset_name=src_dataset_dict['dataset_name'],
                                                set="train",
                                                num_classes=len(src_dataset_dict['classes']),
                                                img_w=opt.img_w, img_h=opt.img_h,
                                                use_augmentation=True,
                                                keep_ratio=False)
    tgt_training_set = dataset.DetectionDataset(root=tgt_dataset_dict['root'], 
                                                dataset_name=tgt_dataset_dict['dataset_name'],
                                                set="train",
                                                num_classes=len(tgt_dataset_dict['classes']),
                                                img_w=opt.img_w, img_h=opt.img_h,
                                                use_augmentation=True,
                                                keep_ratio=False)

    src_training_set_loader = torch.utils.data.DataLoader(src_training_set, 
                                                          opt.batch_size,
                                                          num_workers=opt.num_workers,
                                                          shuffle=True,
                                                          collate_fn=dataset.collate_fn,
                                                          pin_memory=True,
                                                          drop_last=True)
    tgt_training_set_loader = torch.utils.data.DataLoader(tgt_training_set, 
                                                          opt.batch_size,
                                                          num_workers=opt.num_workers,
                                                          shuffle=True,
                                                          collate_fn=dataset.collate_fn,
                                                          pin_memory=True,
                                                          drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    iterations_per_epoch = min(len(src_training_set_loader), len(tgt_training_set_loader))
    total_iteration =  iterations_per_epoch * opt.total_epoch

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)
    
    start_epoch = 0
    if os.path.isfile(opt.weights):
        checkpoint = torch.load(opt.weights)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    for epoch in range(start_epoch, opt.total_epoch):
        model.train()
        
        total_loss = 0.0
        total_dloss1 = 0.0
        total_dloss2 = 0.0
        
        start_steps = epoch * len(src_training_set_loader)
        total_steps = opt.total_epoch * len(tgt_training_set_loader)
        
        for i, (src_batch_data, tgt_batch_data) in enumerate(zip(src_training_set_loader, tgt_training_set_loader)):
            
            src_batch_img = src_batch_data["img"].to(device)
            src_batch_label = src_batch_data["label"]
            
            tgt_batch_img = tgt_batch_data["img"].to(device)
            tgt_batch_label = tgt_batch_data["label"]
            
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            #forward
            # 1. Detection loss
            batch_output, _, _ = model(src_batch_img)
            loss, losses = model.compute_loss(batch_output, src_batch_label)
            
            # 2. Domain loss
            combined_img = torch.cat([src_batch_img, tgt_batch_img], 0)
            _, da_pred1, da_pred2 = model(combined_img, alpha=alpha)
            
            da_label1 = torch.zeros_like(da_pred1).to(device=device)
            da_label1[src_batch_img.shape[0]:, :] = 1.
            
            da_label2 = torch.zeros_like(da_pred2).to(device=device)
            da_label2[src_batch_img.shape[0]:, :] = 1.
            
            domain_loss1 = F.binary_cross_entropy_with_logits(da_pred1, da_label1)
            domain_loss2 = F.binary_cross_entropy_with_logits(da_pred2, da_label2)
            
            loss += opt.lam1 * domain_loss1 + opt.lam2 * domain_loss2

            #backword
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss = loss.item()
            total_dloss1 += domain_loss1.item()
            total_dloss2 += domain_loss2.item()
            
        scheduler.step()
        
        total_loss = total_loss / iterations_per_epoch
        total_dloss1 = total_dloss1 / iterations_per_epoch
        total_dloss2 = total_dloss2 / iterations_per_epoch
        
        print('Epoch [{}/{}] loss={:.6f}, dloss1={:.6f}, dloss2={:.6f}'.format(
            epoch+1, opt.total_epoch, total_loss, total_dloss1, total_dloss2))
        
        checkpoint = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'mAP': 0.,
        'best_mAP': 0.
        }
        
        torch.save(checkpoint, os.path.join(opt.save_folder, 'epoch_' + str(epoch + 1) + '.pth'))
        
