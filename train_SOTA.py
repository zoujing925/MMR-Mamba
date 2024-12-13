"""
Compare with state-of-the-art methods.
Load models from the folder networks/compare_models.
"""

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from skimage import io
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.compare_models import build_model_from_name
from dataloaders.BRATS_dataloader_new import Hybrid as MyDataset
from dataloaders.BRATS_dataloader_new import RandomPadCrop, ToTensor, AddNoise

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from metric import nmse

from utils import bright, trunc

import torchsummary
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xiaohan/datasets/BRATS_dataset/BRATS_2020_images/selected_images/')
parser.add_argument('--MRIDOWN', type=str, default='4X', help='MRI down-sampling rate')
parser.add_argument('--low_field_SNR', type=int, default=15, help='SNR of the simulated low-field image')
parser.add_argument('--phase', type=str, default='train', help='Name of phase')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--exp', type=str, default='msl_model', help='model_name')
parser.add_argument('--max_iterations', type=int, default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0002, help='maximum epoch numaber to train')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

# parser.add_argument('--input_dim', type=int, default=1, help='number of channels of the input image')
# parser.add_argument('--output_dim', type=int, default=1, help='number of channels of the reconstructed image')
parser.add_argument('--model_name', type=str, default='unet_single', help='model_name')
parser.add_argument('--use_multi_modal', type=str, default='False', help='whether use multi-modal data for MRI reconstruction')
parser.add_argument('--modality', type=str, default='t2', help='MRI modality')
parser.add_argument('--input_modality', type=str, default='t2', help='input MRI modality')

parser.add_argument('--relation_consistency', type=str, default='False', help='regularize the consistency of feature relation')

parser.add_argument('--clip_grad', type=str, default='True', help='clip gradient of the network parameters')


parser.add_argument('--norm', type=str, default='False', help='Norm Layer between UNet and Transformer')
parser.add_argument('--input_normalize', type=str, default='mean_std', help='choose from [min_max, mean_std, divide]')

parser.add_argument('--kspace_refine', type=str, default='False', \
                    help='use the original under-sampled input or the kspace-interpolated input')

parser.add_argument('--kspace_round', type=str, default='round4', help='use which round of kspace_recon as model input')



args = parser.parse_args()
train_data_path = args.root_path
test_data_path = args.root_path
snapshot_path = "../model_241005cvpr/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr



def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(
        img1 **2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()



def gradient_calllback(network):
    """
    记录Unet_restormer网络中各层特征参数的gradient.
    """
    for name, param in network.named_parameters():
        if param.grad is not None:
            # print("Gradient of {}: {}".format(name, param.grad.abs().mean()))

            if param.grad.abs().mean() == 0:
                print("Gradient of {} is 0".format(name))

        else:
            print("Gradient of {} is None".format(name))



if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    network = build_model_from_name(args).cuda()
    if len(args.gpu.split(',')) > 1:
        network = nn.DataParallel(network)
        # network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    
    # print("network architecture:", network)
    
    n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('number of params: %.3f M' % (n_parameters / 1024 / 1024), n_parameters)
    
    n_parameters = sum(p.numel() for p in network.parameters())
    print('number of params without requires_grad: %.2f M' % (n_parameters / 1024 / 1024), n_parameters)
    
    # print(torchsummary.summary(network.cuda(), (1, 240, 240)))
    
    # print(parameter_count_table(network))
    # from ptflops import get_model_complexity_info
    # def prepare_input(resolution):
    #     x1 = torch.FloatTensor(1, *resolution)
    #     x2 = torch.FloatTensor(1, *resolution)
    #     return dict(x=x1, complement=x2)
    #     return dict(ref=x1, lr=x2)
    #     return dict(t1=x1, x=x2)
    #     return dict(image1=x1, image2=x2)
    # macs, params = get_model_complexity_info(network, (1, 320, 320), input_constructor=prepare_input,
    #                                          as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # macs, params = get_model_complexity_info(network, (2, 240, 240),
    #                                          as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('FLOPs',macs)
    # print('params',params)

    db_train = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                         split='train', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        #  transform=transforms.Compose([RandomPadCrop(), ToTensor(), AddNoise()]),
                         transform=transforms.Compose([RandomPadCrop(), ToTensor()]),
                         base_dir=train_data_path, input_normalize = args.input_normalize)
    
    db_test = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                        split='test', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    fixtrainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.phase == 'train':
        network.train()

        params = list(network.parameters())
        # optimizer1 = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=1e-4)
        # optimizer1 = optim.Adam(params, lr=base_lr, betas=(0.5, 0.999), weight_decay=1e-4)
        optimizer1 = optim.AdamW(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20000, gamma=0.5)
        # scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=200000, gamma=0.5)

        writer = SummaryWriter(snapshot_path + '/log')

        iter_num = 0
        max_epoch = max_iterations // len(trainloader) + 1
        
        print("max_epoch:", max_epoch, len(trainloader))


        best_status = {'T1_NMSE': 10000000, 'T1_PSNR': 0, 'T1_SSIM': 0,
                       'T2_NMSE': 10000000, 'T2_PSNR': 0, 'T2_SSIM': 0}

        for epoch_num in tqdm(range(max_epoch), ncols=70):
            time1 = time.time()
            for i_batch, (sampled_batch, sample_stats) in enumerate(trainloader):
                time2 = time.time()
                # print("time for data loading:", time2 - time1)

                t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                       sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
                
                # save_path = '../test_input/'
                # t2_img = (t2.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                # io.imsave(save_path + str(i_batch) + '_t2.png', bright(t2_img,0,0.8))
                # t2_in_img = (t2_in.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                # io.imsave(save_path + str(i_batch) + '_t2_in.png', bright(t2_in_img,0,0.8))
                
                # print("t1_in:", t1_in.max(), t1_in.min(), 't1:', t1.max(), t1.min())
                # print("t2_in:", t2_in.max(), t2_in.min(), 't2:', t2.max(), t2.min())

                time3 = time.time()
                if args.use_multi_modal == 'True':
                    
                    if args.modality == "both":
                        # t1_out, t2_out = network(t1_in, t2_in, t1_krecon, t2_krecon)
                        t1_out, t2_out = network(t1_in, t2_in)
                        loss = F.l1_loss(t1_out, t1) + F.l1_loss(t2_out, t2)
                    
                    elif args.modality == "t1":
                        outputs = network(t1_in, t2_in)
                        loss = F.l1_loss(outputs, t1)
                        
                    elif args.modality == "t2":
                        outputs = network(t2_in, t1_in) 
                        # print("recon image:", outputs.shape)
                        loss = F.l1_loss(outputs, t2)


                elif args.use_multi_modal == 'False':
                    if args.modality == "t1":
                        outputs = network(t1_in)
                        # print("t1_recon:", outputs.max(), outputs.min())
                        loss = F.l1_loss(outputs, t1)
                    elif args.modality == "t2":
                        if args.input_modality == "t1":
                            outputs = network(t1_in)
                        elif args.input_modality == "t2":
                            outputs = network(t2_in)
                            
                        # print('t2_out:', outputs.shape, t2.shape)
                        loss = F.l1_loss(outputs, t2)

                    # print("reconstructed image:", outputs.max(), outputs.min())

                time4 = time.time()
                # print("time for network forward:", time4 - time3)

                optimizer1.zero_grad()
                loss.backward()

                if args.clip_grad == "True":
                    ### clip the gradients to a small range.
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.01)

                optimizer1.step()
                scheduler1.step()

                time5 = time.time()
                # print("time for network optimization:", time5 - time4)
                # gradient_calllback(network)
                # print("current learning rate:", scheduler1.get_lr())

                # summary
                iter_num = iter_num + 1
                # writer.add_scalar('lr', scheduler1.get_lr(), iter_num)
                # writer.add_scalar('loss/loss', loss, iter_num)

                if iter_num % 100 == 0:
                    logging.info('iteration %d : learning rate : %f loss : %f ' % (iter_num, scheduler1.get_lr()[0], loss.item()))
    
                if iter_num % 20000 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save({'network': network.state_dict()}, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num > max_iterations:
                    break
                time1 = time.time()
            
            
            ## ================ Evaluate ================
            logging.info(f'Epoch {epoch_num} Evaluation:')
            # print()
            t1_MSE_all, t1_PSNR_all, t1_SSIM_all = [], [], []
            t2_MSE_all, t2_PSNR_all, t2_SSIM_all, t2_NMSE_all, t2_NMSE1_all = [], [], [], [], []

            for (sampled_batch, sample_stats) in testloader:
                
                t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                       sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
                
                if args.use_multi_modal == 'True':
                    ### feed input images to the network.
                    # print('use multi-modal data')
                    if args.modality == "both":
                        t1_out, t2_out = network(t1_in, t2_in)
                    elif args.modality == "t2":
                        t2_out = network(t2_in, t1_in)
                        t1_out = None
                        # print('modality: t2')

                elif args.use_multi_modal == 'False':
                    if args.modality == "t2":
                        # print('use_multi_modal == False, modality: t2')
                        t2_out = network(t2_in)
                        t1_out = None
                        
                        
                # print("t1_in:", t1_in.max(), t1_in.min())
                # print('t2_in:', t2_in.max(), t2_in.min())
                # print("t2_gt:", t2.max(), t2.min())
                # print("t2_output:", t2_out.max(), t2_out.min())
                
                
                if args.input_normalize == "mean_std":
                    # print("mean_std normalization")
                    ### 按照 x*std + mean把图像变回原来的特征范围
                    t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
                    t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
                    t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
                    t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]
                    
                    # print("t1_mean:", t1_mean, "t1_std:", t1_std, "t2_mean:", t2_mean, "t2_std:", t2_std)

                    if t1_out is not None:
                        t1_img = (np.clip(t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                        t1_out_img = (np.clip(t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                        # t1_krecon_img = (np.clip(t1_krecon.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)

 
                    t2_img = (np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                    t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                    # t2_krecon_img = (np.clip(t2_krecon.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)

                    # print("t2 range:", t2_img.max(), t2_img.min())
                    # print("t2recon range:", t2_out_img.max(), t2_out_img.min())

                # else:
                #     if t1_out is not None:
                #         t1_img = (np.clip(t1.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                #         t1_out_img = (np.clip(t1_out.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                #         # t1_krecon_img = (np.clip(t1_krecon.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
 
                #     t2_img = (np.clip(t2.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                #     t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                #     # t2_krecon_img = (np.clip(t2_krecon.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)


                if t1_out is not None:
                    # print("t1_out_img range:", t1_out_img.max(), t1_out_img.min())
                    # print("t1_img range:", t1_img.max(), t1_img.min())
                    MSE = mean_squared_error(t1_img, t1_out_img)
                    PSNR = peak_signal_noise_ratio(t1_img, t1_out_img)
                    SSIM = structural_similarity(t1_img, t1_out_img)
                    t1_MSE_all.append(MSE)
                    t1_PSNR_all.append(PSNR)
                    t1_SSIM_all.append(SSIM)
                    # print("[t1 MRI] MSE:", MSE, "PSNR:", PSNR, "SSIM:", SSIM)

                if t2_out is not None:
                    # print("t2_out_img range:", t2_out_img.max(), t2_out_img.min())
                    # print("t2_img range:", t2_img.max(), t2_img.min())
                    MSE = mean_squared_error(t2_img, t2_out_img)
                    PSNR = peak_signal_noise_ratio(t2_img, t2_out_img)
                    SSIM = structural_similarity(t2_img, t2_out_img)
                    NMSE = nmse(t2_img, t2_out_img)
                    NMSE1 = nmse(t2.data.cpu().numpy()[0, 0], t2_out.data.cpu().numpy()[0, 0])
                    t2_MSE_all.append(MSE)
                    t2_PSNR_all.append(PSNR)
                    t2_SSIM_all.append(SSIM)
                    t2_NMSE_all.append(NMSE)
                    t2_NMSE1_all.append(NMSE1)
                    # print("[t2 MRI] MSE:", MSE, "PSNR:", PSNR, "SSIM:", SSIM)                   

            if t1_out is not None:
                t1_mse = np.array(t1_MSE_all).mean()
                t1_psnr = np.array(t1_PSNR_all).mean()
                t1_ssim = np.array(t1_SSIM_all).mean()

            t2_mse = np.array(t2_MSE_all).mean()
            t2_psnr = np.array(t2_PSNR_all).mean()
            t2_ssim = np.array(t2_SSIM_all).mean()
            t2_nmse = np.array(t2_NMSE_all).mean()
            t2_nmse1 = np.array(t2_NMSE1_all).mean()
            
            t2_mse_std = np.array(t2_MSE_all).std()
            t2_psnr_std = np.array(t2_PSNR_all).std()
            t2_ssim_std = np.array(t2_SSIM_all).std()
            t2_nmse_std = np.array(t2_NMSE_all).std()
            t2_nmse1_std = np.array(t2_NMSE1_all).std()
            
            if t2_psnr > best_status['T2_PSNR']:
                if args.modality == "both":
                    best_status = {'T1_NMSE': t1_mse, 'T2_PSNR': t1_psnr, 'T2_SSIM': t1_ssim,
                                'T2_NMSE': t2_mse, 'T2_PSNR': t2_psnr, 'T2_SSIM': t2_ssim}
                elif args.modality == "t2":
                    best_status = {'T2_NMSE': t2_mse, 'T2_PSNR': t2_psnr, 'T2_SSIM': t2_ssim}

                best_checkpoint_path = os.path.join(snapshot_path, 'best_checkpoint.pth')

                torch.save({'network': network.state_dict()}, best_checkpoint_path)
                print('New Best Network:')
            if args.modality == "both":
                logging.info(f"[T1 MRI:] average MSE: {t1_mse} average PSNR: {t1_psnr} average SSIM: {t1_ssim}")
            logging.info(f"[T2 MRI:] average MSE: {t2_mse} average PSNR: {t2_psnr} average SSIM: {t2_ssim} average NMSE: {t2_nmse}, average NMSE1: {t2_nmse1}")
            logging.info(f"[T2 MRI:] std MSE: {t2_mse_std} std PSNR: {t2_psnr_std} std SSIM: {t2_ssim_std} std NMSE: {t2_nmse_std}, std NMSE: {t2_nmse1_std}")

            if iter_num > max_iterations:
                break
        print(best_status)
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save({'network': network.state_dict()},
                   save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        writer.close()
