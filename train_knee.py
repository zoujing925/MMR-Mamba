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
from dataloaders.Knee_dataloader import KneeDataset as MyDataset
from dataloaders.Knee_dataloader import RandomPadCrop, ToTensor, AddNoise

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict

from utils import bright, trunc
from matplotlib import pyplot as plt
from thop import profile

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


parser.add_argument('--MASKTYPE', type=str, default='random', help='mask type for kspace sampling') # "random" or "equispaced"
parser.add_argument('--CENTER_FRACTIONS', type=list, default=[0.04], help='center fraction for kspace sampling')
parser.add_argument('--ACCELERATIONS', type=list, default=[8], help='acceleration for kspace sampling')


args = parser.parse_args()
train_data_path = args.root_path
test_data_path = args.root_path
snapshot_path = "../model_241005cvpr_knee/" + args.exp + "/"

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
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))
    
    input = torch.randn(1, 1, 320, 320).cuda()
    ref = torch.randn(1, 1, 320, 320).cuda()
    flops, params = profile(network, inputs=(input, ref))
    print('FLOPs:', flops/1024**3, "G", 'params:', params/1024**2, "M")

    db_train = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                         split='train', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                         transform=transforms.Compose([RandomPadCrop(), ToTensor()]),
                         base_dir=train_data_path, input_normalize = args.input_normalize,
                         CENTER_FRACTIONS=args.CENTER_FRACTIONS, ACCELERATIONS=args.ACCELERATIONS)
    
    db_test = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                        split='val', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize,
                        CENTER_FRACTIONS=args.CENTER_FRACTIONS, ACCELERATIONS=args.ACCELERATIONS)

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
            for i_batch, (pd, pdfs, _) in enumerate(trainloader):
                time2 = time.time()
                # print("time for data loading:", time2 - time1)
                                       
                t1_in, t1, t2_in, t2 = pd[1].unsqueeze(1).cuda(), pd[1].unsqueeze(1).cuda(), \
                                        pdfs[0].unsqueeze(1).cuda(), pdfs[1].unsqueeze(1).cuda()           
                # t1_in, t1, t2_in, t2 = pdfs[1].unsqueeze(1).cuda(), pdfs[1].unsqueeze(1).cuda(), \
                                        # pdfs[0].unsqueeze(1).cuda(), pdfs[1].unsqueeze(1).cuda()        

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

                ################# add frequency loss ##################
                # _, (outputs_LL, outputs_HL, outputs_LH, outputs_HH) = dwt_init(outputs)
                # _, (t2_LL, t2_HL, t2_LH, t2_HH) = dwt_init(t2)
                # loss_hf = F.l1_loss(outputs_HL, t2_HL) + F.l1_loss(outputs_LH, t2_LH) + F.l1_loss(outputs_HH, t2_HH)
                
                # outputsF = torch.fft.rfft2(outputs+1e-8, norm='backward')
                # t2F = torch.fft.rfft2(t2+1e-8, norm='backward')
                # outputsF_amp = torch.abs(outputsF); t2F_amp = torch.abs(t2F)
                # outputsF_pha = torch.angle(outputsF); t2F_pha = torch.angle(t2F)
                # loss_fft = F.l1_loss(outputsF_amp, t2F_amp) + F.l1_loss(outputsF_pha, t2F_pha)
                
                # lambda_hf = 0.01; lambda_fft = 0.0001
                # loss_all = loss + lambda_hf * loss_hf + lambda_fft * loss_fft
                
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
                    # logging.info('iteration %d : learning rate : %f lossall : %f loss: %f loss_hf: %f loss_fft: %f' % \
                    #     (iter_num, scheduler1.get_lr()[0], loss_all.item(), loss.item(), loss_hf.item()*lambda_hf, loss_fft.item()*lambda_fft))
    
    
                if iter_num % 20000 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save({'network': network.state_dict()}, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num > max_iterations:
                    break
                time1 = time.time()
            
            
            ## ================ Evaluate ================
            logging.info(f'Epoch {epoch_num} Evaluation:')
                  
            output_dic = defaultdict(dict)
            target_dic = defaultdict(dict)
            # input_dic = defaultdict(dict)
            nmse_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            # print()
            t1_MSE_all, t1_PSNR_all, t1_SSIM_all = [], [], []
            t2_MSE_all, t2_PSNR_all, t2_SSIM_all = [], [], []

            for (pd, pdfs, _) in testloader:
                
                t1_in, t1, t2_in, t2 = pd[1].unsqueeze(1).cuda(), pd[1].unsqueeze(1).cuda(), \
                                        pdfs[0].unsqueeze(1).cuda(), pdfs[1].unsqueeze(1).cuda()   
                
                mean = pdfs[2].unsqueeze(1).unsqueeze(2).cuda()
                std = pdfs[3].unsqueeze(1).unsqueeze(2).cuda()

                fname = pdfs[4]
                slice_num = pdfs[5]               
                name = os.path.basename(pdfs[4][0]).split('.')[0]

                with torch.no_grad():
                    t2_out = network(t2_in, t1_in)
                    t1_out = None
                
                outputs = t2_out[0].squeeze(1)
                target = t2.squeeze(1)
                # print('output before normalization:', outputs.max(), outputs.min())
                         
                outputs = outputs * std + mean
                # print("outputs:", outputs.max(), outputs.min())
                target = target * std + mean
                inputs = t2_in.squeeze(1) * std + mean
                
                # print('target:', target.max(), target.min())
                # print('output:', outputs.max(), outputs.min())
                
                # fig = plt.figure()
                # plt.axis('off')
                # plt.imshow(outputs[0].cpu().numpy(), cmap = 'gray')
                # fig.savefig('./test_images/output.png', bbox_inches='tight', pad_inches=0)

                output_dic[fname[0]][slice_num[0]] = outputs[0]
                target_dic[fname[0]][slice_num[0]] = target[0]
                # input_dic[fname[0]][slice_num[0]] = inputs[0]
                        
                # our_nmse = nmse(target[0].cpu().numpy(), outputs[0].cpu().numpy())
                # our_psnr = psnr(target[0].cpu().numpy(), outputs[0].cpu().numpy())
                # our_ssim = ssim(target[0].cpu().numpy(), outputs[0].cpu().numpy())
                    
                # our_nmse = nmse(target[0].detach().cpu().numpy(), outputs[0].detach().cpu().numpy())
                # our_psnr = psnr(target[0].detach().cpu().numpy(), outputs[0].detach().cpu().numpy())
                # our_ssim = ssim(target[0].detach().cpu().numpy(), outputs[0].detach().cpu().numpy())
                    
                # print('name:{}, slice:{}, psnr:{}, ssim:{}'.format(name, slice_num[0], our_psnr, our_ssim))
                # breakpoint()  
                
            # t2_MSE_all.append(our_nmse)
            # t2_PSNR_all.append(our_psnr)
            # t2_SSIM_all.append(our_ssim)
            # print('Evaluation by slice:', 'MSE:', np.array(t2_MSE_all).mean(), 'PSNR:', np.array(t2_PSNR_all).mean(), 'SSIM:', np.array(t2_SSIM_all).mean()) 

            for name in output_dic.keys():
                f_output = torch.stack([v for _, v in output_dic[name].items()])
                f_target = torch.stack([v for _, v in target_dic[name].items()])
                our_nmse = nmse(f_target.detach().cpu().numpy(), f_output.detach().cpu().numpy())
                our_psnr = psnr(f_target.detach().cpu().numpy(), f_output.detach().cpu().numpy())
                our_ssim = ssim(f_target.detach().cpu().numpy(), f_output.detach().cpu().numpy())
                # print('name:{}, psnr:{}, ssim:{}'.format(name, our_psnr, our_ssim))
                nmse_meter.update(our_nmse, 1)
                psnr_meter.update(our_psnr, 1)
                ssim_meter.update(our_ssim, 1)
            

            print("==> Evaluate Metric")
            print("Results ----------")
            # print("NMSE: {:.4}".format(nmse_meter.avg))
            # print("PSNR: {:.4}".format(psnr_meter.avg))
            # print("SSIM: {:.4}".format(ssim_meter.avg))
            print('mean',"NMSE: {:.4}".format(np.array(nmse_meter.score).mean()),"PSNR: {:.4}".format(np.array(psnr_meter.score).mean()),"SSIM: {:.4}".format(np.array(ssim_meter.score).mean()))
            print('std',"NMSE: {:.4}".format(np.array(nmse_meter.score).std()),"PSNR: {:.4}".format(np.array(psnr_meter.score).std()),"SSIM: {:.4}".format(np.array(ssim_meter.score).std()))
            print("------------------")
            
            # return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}                  

            t2_nmse = np.array(nmse_meter.score).mean()
            t2_psnr = np.array(psnr_meter.score).mean()
            t2_ssim = np.array(ssim_meter.score).mean()
            
            t2_nmse_std = np.array(nmse_meter.score).std()
            t2_psnr_std = np.array(psnr_meter.score).std()
            t2_ssim_std = np.array(ssim_meter.score).std()
            
            if t2_psnr > best_status['T2_PSNR']:
                
                best_status = {'T2_NMSE': t2_nmse, 'T2_PSNR': t2_psnr, 'T2_SSIM': t2_ssim}

                best_checkpoint_path = os.path.join(snapshot_path, 'best_checkpoint.pth')

                torch.save({'network': network.state_dict()}, best_checkpoint_path)
                print('New Best Network:')
            logging.info(f"[T2 MRI:] average NMSE: {t2_nmse}, std: {t2_nmse_std} average PSNR: {t2_psnr}, std: {t2_psnr_std}, average SSIM: {t2_ssim}, std: {t2_ssim_std}")
            

            if iter_num > max_iterations:
                break
        print(best_status)
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save({'network': network.state_dict()},
                   save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        writer.close()
