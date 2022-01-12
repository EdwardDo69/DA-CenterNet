from models import centernet
from utils import common
from data import dataset

import torch
import torch.cuda.amp
from torch.utils.tensorboard import SummaryWriter

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for forward')
    parser.add_argument('--lr', default=0.0008, type=float, help='Learning rate')
    parser.add_argument('--milestone', default=[40], type=list, help='Milestones for learning rate scheduler')
    
    parser.add_argument('--img-w', default=512, type=int)
    parser.add_argument('--img-h', default=512, type=int)

    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--total-epoch', type=int, default=70, help='total_epoch')

    parser.add_argument('--data', type=str, default="./data/voc0712.yaml")
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--save-folder', default='./weights', type=str, help='where you save weights')
    parser.add_argument('--seed', default=7777, type=int)

    opt = parser.parse_args()
    common.setup_seed(opt.seed)
    common.mkdir(dir=opt.save_folder, remove_existing_dir=False)
    
    dataset_dict = common.parse_yaml(opt.data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = centernet.CenterNet(num_classes=len(dataset_dict['classes']), pretrained_backbone=True)
    model = model.to(device=device)
    
    training_set = dataset.DetectionDataset(root=dataset_dict['root'], 
                                            dataset_name=dataset_dict['dataset_name'],
                                            set="train",
                                            num_classes=len(dataset_dict['classes']),
                                            img_w=opt.img_w, img_h=opt.img_h,
                                            use_augmentation=True,
                                            keep_ratio=False)

    training_set_loader = torch.utils.data.DataLoader(training_set, 
                                                      opt.batch_size,
                                                      num_workers=opt.num_workers,
                                                      shuffle=True,
                                                      collate_fn=dataset.collate_fn,
                                                      pin_memory=True,
                                                      drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    iterations_per_epoch = len(training_set_loader)
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
        
        for i, batch_data in enumerate(training_set_loader):
            
            n_iteration = (iterations_per_epoch * epoch) + i
            
            batch_img = batch_data["img"].to(device)
            batch_label = batch_data["label"]
            
            #forward
            batch_output = model(batch_img)
            loss, losses = model.compute_loss(batch_output, batch_label)

            #backword
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss = loss.item()
            
        scheduler.step()
        
        total_loss = total_loss / iterations_per_epoch
        
        print('Epoch [{}/{}] loss={:.6f}'.format(epoch+1, opt.total_epoch, total_loss))
        
        checkpoint = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'mAP': 0.,
        'best_mAP': 0.
        }
        
        torch.save(checkpoint, os.path.join(opt.save_folder, 'epoch_' + str(epoch + 1) + '.pth'))
        
