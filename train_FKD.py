import time
import argparse
import datetime
import os
import distiller
import math
from cv2 import threshold

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage  
from tensorboardX import SummaryWriter
from typing import OrderedDict
from model import PTModel
from loss import ssim
from data import getTrainingTestingData
from utils2 import AverageMeter, DepthNorm, colorize
from scipy.stats import norm
from loss import masked_l1_loss
from matplotlib import pyplot as plt

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--model_folder', type=str, default='model_mobilenetv2_MKD_up_50000')
    parser.add_argument('--pretrained_model', default='models/model_mobilenetv2_NT_pretrained_50000/model_7.pth', type=str)
    parser.add_argument('--pretrained_model2', default='', type=str)
    args = parser.parse_args()

    # hyperparameter
    # alpha = 0.9
    # T = 20

    # Create model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_s = nn.DataParallel(PTModel().cuda()).to(device)
    model_t = nn.DataParallel(PTModel().cuda()).to(device)
    print('Model created.')

    state_dict = torch.load(args.pretrained_model)
    # state_dict2 = torch.load(args.pretrained_model2)
    model_t.load_state_dict(state_dict)
    
    model_KD = distiller.Distiller(model_t, model_s)
    model_KD = nn.DataParallel(model_KD).to(device)
    # model_KD.load_state_dict(state_dict2)

    folder_name = 'models/' + args.model_folder
    if not os.path.exists(folder_name): os.mkdir(folder_name)

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model_s.module.get_1x_lr_params(), 'lr': args.lr},
                                   {'params': model_s.module.get_10x_lr_params(), 'lr': args.lr * 10},
                                   {'params': model_KD.module.update_mask.parameters(), 'lr': args.lr},
                                   {'params': model_KD.module.Connectors.parameters(), 'lr': args.lr * 10}],
                                  weight_decay=1e-2, eps=1e-3)
    # optimizer = torch.optim.Adam( model_KD.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)
    trans = ToPILImage()

    # Loss
    l1_criterion = masked_l1_loss()
    # l1_criterion = nn.L1Loss(reduction='mean')
    # l1_criterion_kd = nn.L1Loss()

    
    # Start training...

    for epoch in range(0, args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode

        model_KD.train()
        model_KD.module.t_net.train()
        model_KD.module.s_net.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda(device))
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(device, non_blocking=True))
            image_NT = torch.autograd.Variable(sample_batched['image_NT'].cuda(device))
            mask_bb = torch.autograd.Variable(sample_batched['mask'].cuda(device))
            # mask_depth = torch.autograd.Variable(sample_batched['mask_depth'].cuda(device))
            mask_depth = (depth < 50)
            mask_bb = mask_bb * mask_depth
            # Normalize depth

            # Masking tensor for missing depth value
            # Predict
            output, loss_distill = model_KD(image, image_NT, mask_bb)
            #teacher_output = model_KD(image)
            
            # FKD loss

            # KD loss
            #KD_loss = l1_criterion_kd(output, t_output)

            # Compute the loss
            l_depth = l1_criterion.forward(output[mask_depth], depth[mask_depth])
            if output.max() > 1000:
                plt.imshow(image[0,0,:,:].cpu().detach().numpy())
                plt.savefig('test.png')
                print(i)
            # l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            # loss = (1.0 * l_ssim) + (0.1 * l_depth)
            #loss = ((1 - alpha) * l_depth) + (KD_loss * alpha)
            #loss = ((1 - alpha) * l_depth) + (KD_loss * alpha) + (loss_distill.sum() / batch_size * 1e-5)
            loss = (l_depth*0.1) + (loss_distill.sum() / batch_size * 1e-8)
            
            if np.isnan(l_depth.cpu().detach().numpy()) == True:
                print(loss)
                print(i)
            if loss > 100:
                print(loss)
            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 50 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 300 == 0:
                LogProgress(model_KD, writer, test_loader, niter, device)

        # Record epoch's intermediate results
        LogProgress(model_KD, writer, test_loader, niter, device)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        if epoch % 1 == 0:
            validate_Unet(epoch, model_KD, device, test_loader, l1_criterion)
        if epoch % 1 == 0:
            model_file = folder_name + "/model_" + str(epoch) + ".pth"    
            torch.save(model_KD.state_dict(), model_file)

def validate_Unet(epoch, model, device, test_loader, l1_criterion):
    model.eval()
    validation_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    delta1_accuracy_bb = 0
    delta2_accuracy_bb = 0
    delta3_accuracy_bb = 0
    rmse_linear_loss_bb = 0
    rmse_log_loss_bb = 0
    abs_relative_difference_loss_bb = 0
    squared_relative_difference_loss_bb = 0
    print("Epochs:      Val_loss    Delta_1     Delta_2     Delta_3    RMSE     RMSE_log    abs_rel.  square_relative")
    with torch.no_grad():
        for idx, image in enumerate(test_loader):
            # pdb.set_trace()
            '''
            x = image['image'].cuda(device)
            y = image['depth'].cuda(device)
            '''
            
            x = torch.autograd.Variable(image['image'].cuda(device))
            x_nt = torch.autograd.Variable(image['image_NT'].cuda(device))
            y = torch.autograd.Variable(image['depth'].cuda(device, non_blocking=True))
            mask_bb = torch.autograd.Variable(image['mask'].cuda(device))
            # mask_depth = torch.autograd.Variable(image['mask_depth'].cuda(device))
            # a = nn.L1Loss()
        
            y_hat, _ = model(x)
            mask_depth = (y <50)
            
            mask_bb = (mask_bb * mask_depth).bool()
            y = DepthNorm(y)
            y_hat = DepthNorm(y_hat)
            y_hat = y_hat*32768/1000000
            y = y*32768/1000000

            loss = l1_criterion.forward(y_hat[mask_depth], y[mask_depth]).mean()
            # l_depth = a(y_hat[mask_depth], y[mask_depth])
            # l_ssim = torch.clamp((1 - ssim(y_hat, y, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            # loss = (1.0 * l_ssim) + (0.1 * l_depth)
            # loss = (l_depth)
            validation_loss += loss
            # mask_depth = mask_depth.float()

            # all error functions
            delta1_accuracy += threeshold_percentage(y_hat, y, mask_depth, 1.05)
            delta2_accuracy += threeshold_percentage(y_hat, y, mask_depth, 1.1)
            delta3_accuracy += threeshold_percentage(y_hat, y, mask_depth, 1.25)
            rmse_linear_loss += rmse_linear(y_hat, y, mask_depth)
            rmse_log_loss += rmse_log(y_hat, y, mask_depth)
            abs_relative_difference_loss += abs_relative_difference(y_hat, y, mask_depth)
            squared_relative_difference_loss += squared_relative_difference(y_hat, y, mask_depth)
            delta1_accuracy_bb += threeshold_percentage(y_hat, y, mask_bb, 1.05)
            delta2_accuracy_bb += threeshold_percentage(y_hat, y, mask_bb, 1.1)
            delta3_accuracy_bb += threeshold_percentage(y_hat, y, mask_bb, 1.25)
            rmse_linear_loss_bb += rmse_linear(y_hat, y, mask_bb)
            rmse_log_loss_bb += rmse_log(y_hat, y, mask_bb)
            abs_relative_difference_loss_bb += abs_relative_difference(y_hat, y, mask_bb)
            squared_relative_difference_loss_bb += squared_relative_difference(y_hat, y, mask_bb)
        validation_loss /= (idx + 1)
        delta1_accuracy /= (idx + 1)
        delta2_accuracy /= (idx + 1)
        delta3_accuracy /= (idx + 1)
        rmse_linear_loss /= (idx + 1)
        rmse_log_loss /= (idx + 1)
        abs_relative_difference_loss /= (idx + 1)
        squared_relative_difference_loss /= (idx + 1)
        delta1_accuracy_bb /= (idx + 1)
        delta2_accuracy_bb /= (idx + 1)
        delta3_accuracy_bb /= (idx + 1)
        rmse_linear_loss_bb /= (idx + 1)
        rmse_log_loss_bb /= (idx + 1)
        abs_relative_difference_loss_bb /= (idx + 1)
        squared_relative_difference_loss_bb /= (idx + 1)
        print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.
            format(epoch, validation_loss, delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss, rmse_log_loss, 
            abs_relative_difference_loss, squared_relative_difference_loss))

        print("Epo_bb:      Val_loss    Delta_1     Delta_2     Delta_3    RMSE     RMSE_log    abs_rel.  square_relative")
        print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.
            format(epoch, validation_loss, delta1_accuracy_bb, delta2_accuracy_bb, delta3_accuracy_bb, rmse_linear_loss_bb, rmse_log_loss_bb, 
            abs_relative_difference_loss_bb, squared_relative_difference_loss_bb))

def LogProgress(model, writer, test_loader, epoch, device):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda(device))
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(device, non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image)[0] )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output

def threeshold_percentage(output, target, mask, threeshold_val):
    output = output[mask]
    target = target[mask]
    d1 = output/target
    d2 = target/output
    max_d1_d2 = torch.max(d1,d2)
    zero = torch.zeros(output.shape[0])
    one = torch.ones(output.shape[0])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero).cuda()
    threeshold_mat = bit_mat.sum() / len(bit_mat)
    return threeshold_mat.mean()

def rmse_linear(output, target, mask):
    output = output[mask]
    target = target[mask]
    diff = output - target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2) / len(diff2)
    rmse = torch.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target, mask):
    output = output[mask]
    target = target[mask]
    diff_log = torch.log(output) - torch.log(target)
    diff2_log = torch.pow(diff_log, 2)
    mse_log = torch.sum(diff2_log) / len(diff2_log)
    rmse_log = torch.sqrt(mse_log)
    return rmse_log.mean()

def abs_relative_difference(output, target, mask):
    output = output[mask]
    target = target[mask]
    abs_relative_diff = torch.abs(output - target) / (target)
    abs_relative_diff = torch.sum(abs_relative_diff) / len(abs_relative_diff)
    return abs_relative_diff.mean()

def squared_relative_difference(output, target, mask):
    output = output[mask]
    target = target[mask]
    square_relative_diff = torch.pow(torch.abs(output - target), 2) / (target)
    square_relative_diff = torch.sum(square_relative_diff) / len(square_relative_diff)
    return square_relative_diff.mean()

if __name__ == '__main__':
    main()
