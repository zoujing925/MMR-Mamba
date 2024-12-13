import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import numpy as np
from skimage import io
from scipy.ndimage import zoom
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from networks.compare_models import build_model_from_name
from networks.wavelet import build_model_from_name
from dataloaders.BRATS_dataloader_new import Hybrid as MyDataset
from dataloaders.BRATS_dataloader_new import ToTensor
from plot_learnablefilter import visualize_filter

# from dataloaders.BRATS_dataloader import Hybrid as MyDataset
# from dataloaders.BRATS_dataloader import ToTensor

from utils import bright, trunc

### Xiaohan, add evaluation metrics
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from metric import nmse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xiaohan/datasets/BRATS_dataset/BRATS_2020_images/selected_images/')
parser.add_argument('--MRIDOWN', type=str, default='4X', help='MRI down-sampling rate')
parser.add_argument('--low_field_SNR', type=int, default=15, help='SNR of the simulated low-field image')
parser.add_argument('--phase', type=str, default='test', help='Name of phase')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--exp', type=str, default='msl_model', help='model_name')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--base_lr', type=float, default=0.0002, help='maximum epoch numaber to train')

# parser.add_argument('--input_dim', type=int, default=1, help='number of channels of the input image')
# parser.add_argument('--output_dim', type=int, default=1, help='number of channels of the reconstructed image')
parser.add_argument('--model_name', type=str, default='unet_single', help='model_name')
parser.add_argument('--use_multi_modal', type=str, default='False', help='whether use multi-modal data for MRI reconstruction')
parser.add_argument('--modality', type=str, default='t2', help='MRI modality')
parser.add_argument('--input_modality', type=str, default='t2', help='input MRI modality')

parser.add_argument('--relation_consistency', type=str, default='False', help='regularize the consistency of feature relation')

parser.add_argument('--norm', type=str, default='False', help='Norm Layer between UNet and Transformer')
parser.add_argument('--input_normalize', type=str, default='mean_std', help='choose from [min_max, mean_std, divide]')
parser.add_argument('--kspace_refine', type=str, default='False', \
                    help='use the original under-sampled input or the kspace-interpolated input')

parser.add_argument('--kspace_round', type=str, default='round4', help='use which round of kspace_recon as model input')


args = parser.parse_args()
test_data_path = args.root_path
snapshot_path = "../model_241005cvpr/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def normalize_output(out_img):
    out_img = (out_img - out_img.min())/(out_img.max() - out_img.min() + 1e-8)
    return out_img


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
    
    # print("network architecture:", network)
    # for name, param in network.named_parameters():
    #     print(name)

    db_test = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round, 
                        split='test', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.phase == 'test':

        # save_mode_path = os.path.join(snapshot_path, 'iter_100000.pth')
        save_mode_path = os.path.join(snapshot_path, 'best_checkpoint.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'], strict=False)
        network.eval()
        cnt = 0
        save_path = snapshot_path + '/result_case/'
        feature_save_path = snapshot_path + '/feature_visualization/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)

        # for name, param in network.named_parameters():
        #     print(name, param)

        t1_MSE_all, t1_PSNR_all, t1_SSIM_all = [], [], []
        t2_MSE_all, t2_PSNR_all, t2_SSIM_all, t2_NMSE_all = [], [], [], []

        for (sampled_batch, sample_stats) in tqdm(testloader, ncols=70):
            cnt += 1
            print('processing ' + str(cnt) + ' image')
            t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                   sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
            # t1_krecon, t2_krecon = sampled_batch['image_krecon'].cuda(), sampled_batch['target_krecon'].cuda()

            # t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
            # t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
            # t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
            # t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]

            # t1 = t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean


            # t1_img = (t1.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            # # io.imsave(save_path + str(cnt) + '_t1.png', bright(t1_img,0,0.8))

            # t1_in_img = (t1_in.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            # # io.imsave(save_path + str(cnt) + '_t1_in.png', bright(t1_in_img,0,0.8))

            # t2_img = (t2.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            # io.imsave(save_path + str(cnt) + '_t2.png', bright(t2_img,0,0.8))

            # t2_in_img = (t2_in.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            # io.imsave(save_path + str(cnt) + '_t2_in.png', bright(t2_in_img,0,0.8))

            t1_out, t2_out = None, None

            if args.use_multi_modal == 'True':
                if args.modality == "both":
                    # t1_out, t2_out = network(t1_in, t2_in)
                    t1_out, t2_out = network(t1_in, t2_in)

                    # print("t1 output:", t1_out.max(), t1_out.min(), t1_out.mean())
                    # print("t1 ground-truth:", t1.max(), t1.min(), t1.mean())
                    # print("t2 output:", t2_out.max(), t2_out.min(), t2_out.mean())
                    # print("t2 ground-truth:", t2.max(), t2.min(), t2.mean())
                    
                    # t1_out, t2_out, feature_stack1, feature_stack2 = network(t1_in, t2_in)
                    # for layer in range(len(feature_stack1)):
                    #     feature1 = np.squeeze(np.mean(np.clip(feature_stack1[layer].detach().cpu().numpy(), a_min=0, a_max=None), 1))
                    #     feature2 = np.squeeze(np.mean(np.clip(feature_stack2[layer].detach().cpu().numpy(), a_min=0, a_max=None), 1))
                    #     feature1_img = (255*(feature1 - feature1.min())/(feature1.max() - feature1.min())).astype(np.uint8)
                    #     feature2_img = (255*(feature2 - feature2.min())/(feature2.max() - feature2.min())).astype(np.uint8)
                    #     scale = (240//feature1_img.shape[0], 240//feature1_img.shape[1])
                    #     feature1_img = zoom(feature1_img, scale, order=2)
                    #     feature2_img = zoom(feature2_img, scale, order=2)
                    #     io.imsave(feature_save_path + str(cnt) + '_t1_layer' + str(layer+1) + '.png', feature1_img)
                    #     io.imsave(feature_save_path + str(cnt) + '_t2_layer' + str(layer+1) + '.png', feature2_img)

                elif args.modality == "t1":
                    t1_out = network(t1_in, t2_in)
                elif args.modality == "t2":
                    t2_out = network(t2_in, t1_in)
                    # t2_out = t2_in

                    # t2_out, unet_features, trans_features = network(t2_in, t1_in)

                    # for layer in range(len(unet_features)):
                    #     feature1 = np.squeeze(np.mean(np.clip(unet_features[layer].detach().cpu().numpy(), a_min=0, a_max=None), 1))
                    #     feature2 = np.squeeze(np.mean(np.clip(trans_features[layer].detach().cpu().numpy(), a_min=0, a_max=None), 1))
                    #     feature1_img = (255*(feature1 - feature1.min())/(feature1.max() - feature1.min())).astype(np.uint8)
                    #     feature2_img = (255*(feature2 - feature2.min())/(feature2.max() - feature2.min())).astype(np.uint8)
                    #     scale = (240//feature1_img.shape[0], 240//feature1_img.shape[1])
                    #     feature1_img = zoom(feature1_img, scale, order=2)
                    #     feature2_img = zoom(feature2_img, scale, order=2)
                    #     # print("[ART layer] input and output feature difference:", np.mean(np.abs(feature1_img - feature2_img)))
                        
                    #     io.imsave(feature_save_path + str(cnt) + '_unet_layer' + str(layer+1) + '.png', feature1_img)
                    #     io.imsave(feature_save_path + str(cnt) + '_trans_layer' + str(layer+1) + '.png', feature2_img)


                    # t2_out, unet_features = network(t2_in, t1_in)

                    # for layer in range(len(unet_features)):
                    #     feature1 = np.squeeze(np.mean(np.clip(unet_features[layer].detach().cpu().numpy(), a_min=0, a_max=None), 1))
                    #     feature1_img = (255*(feature1 - feature1.min())/(feature1.max() - feature1.min())).astype(np.uint8)
                    #     scale = (240//feature1_img.shape[0], 240//feature1_img.shape[1])
                    #     feature1_img = zoom(feature1_img, scale, order=2)
                    #     # print("[ART layer] input and output feature difference:", np.mean(np.abs(feature1_img - feature2_img)))
                    #     io.imsave(feature_save_path + str(cnt) + '_unet_layer' + str(layer+1) + '.png', feature1_img)



                if args.input_normalize == "mean_std":
                    ### 按照 x*std + mean把图像变回原来的特征范围
                    t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
                    t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
                    t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
                    t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]

                    # t1_img = ((t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean) * 255).astype(np.uint8)
                    # t2_img = ((t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean) * 255).astype(np.uint8)
                    # t1_out_img = ((t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean) * 255).astype(np.uint8)
                    # t2_out_img = ((t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean) * 255).astype(np.uint8)

                    # t1_img = (np.clip(t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                    t2_img = (np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                    t2_in_img = (np.clip(t2_in.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                    t1_in_img = (np.clip(t1_in.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                    # t1_out_img = (np.clip(t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                    t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)

                    # print("t1_img range:", t1_img.max(), t1_img.min())
                    # print("t1_out_img range:", t1_out_img.max(), t1_out_img.min())
                    # print("t2_img range:", t2_img.max(), t2_img.min())
                    # print("t2_out_img range:", t2_out_img.max(), t2_out_img.min())

                    if cnt > 1020 and cnt < 1050:
                        save_path = snapshot_path + '/result_case_best/' + 'diff/'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)   
                        
                        io.imsave(save_path + str(cnt) + '_t2_gt.png', bright(t2_img, 0, 0.8))
                        io.imsave(save_path + str(cnt) + '_t2_out.png', bright(t2_out_img, 0, 0.8))
                        io.imsave(save_path + str(cnt) + '_t2_in.png', bright(t2_in_img, 0, 0.8))

                        diff_t2 = (trunc(t2_img - t2_out_img +135)).astype(np.uint8)
                        io.imsave(save_path + str(cnt) + '_t2_diff.png', diff_t2)
                        
                        # diff = np.abs(np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) - np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1)) * 255
                        diff = np.abs((t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean) - (t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean)) * 255
                        print("diff range:", diff.max(), diff.min())
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(diff, cmap='jet', vmin=0, vmax=50)
                        plt.savefig(save_path + str(cnt) + '_error_map50.png', bbox_inches='tight',pad_inches = 0)    
                        
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(diff, cmap='jet', vmin=0, vmax=40)
                        plt.savefig(save_path + str(cnt) + '_error_map40.png', bbox_inches='tight',pad_inches = 0)
                        
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(diff, cmap='jet', vmin=0, vmax=30)
                        plt.savefig(save_path + str(cnt) + '_error_map30.png', bbox_inches='tight',pad_inches = 0)  
                        
                        diff_origin = np.abs((t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean) - (t2_in.data.cpu().numpy()[0, 0] * t2_std + t2_mean)) * 255
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(diff_origin, cmap='jet', vmin=0, vmax=50)
                        plt.savefig(save_path + str(cnt) + '_error_map_origin50.png', bbox_inches='tight',pad_inches = 0)               
                    
                    if cnt == 9:
                        ############################## plot filter ##############################
                        save_path_filter = snapshot_path + '/result_case_best_filter_coolwarm0.6/'
                        if not os.path.exists(save_path_filter):
                            os.makedirs(save_path_filter)   
                        visualize_filter(network.wfca1.filter_hl, save_path_filter + str(cnt) +'_wfca1_' + '_filterhl.png')
                        visualize_filter(network.wfca1.filter_lh, save_path_filter + str(cnt) +'_wfca1_' + '_filterlh.png')
                        visualize_filter(network.wfca1.filter_hh, save_path_filter + str(cnt) +'_wfca1_' + '_filterhh.png')
                        visualize_filter(network.wfca2.filter_hl, save_path_filter + str(cnt) +'_wfca2_' + '_filterhl.png')
                        visualize_filter(network.wfca2.filter_lh, save_path_filter + str(cnt) +'_wfca2_' + '_filterlh.png')
                        visualize_filter(network.wfca2.filter_hh, save_path_filter + str(cnt) +'_wfca2_' + '_filterhh.png')
                        visualize_filter(network.wfca3.filter_hl, save_path_filter + str(cnt) +'_wfca3_' + '_filterhl.png')
                        visualize_filter(network.wfca3.filter_lh, save_path_filter + str(cnt) +'_wfca3_' + '_filterlh.png')
                        visualize_filter(network.wfca3.filter_hh, save_path_filter + str(cnt) +'_wfca3_' + '_filterhh.png')

                # if t2_out is not None:
                #     t2_out[t2_out < 0.0] = 0.0
                #     # print("t2_recon range:", t2_out.max(), t2_out.min())
                #     t2_out_img = (t2_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                #     io.imsave(save_path + str(cnt) + '_t2_out.png', bright(t2_out_img,0,0.8))
                    
                #     diff_t2 = t2_out.data.cpu().numpy()[0, 0] - t2.data.cpu().numpy()[0, 0]
                #     diff_t2 = (trunc(diff_t2*255 +135)).astype(np.uint8)
                #     io.imsave(save_path + str(cnt) + '_t2_diff.png', diff_t2)                    
                
                
            elif args.use_multi_modal == 'False':
                if args.modality == "t1":
                    t1_out = network(t1_in)
                    t1_out[t1_out < 0.0] = 0.0
                    t1_out_img = (t1_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                    io.imsave(save_path + str(cnt) + '_t1_out.png', bright(t1_out_img,0,0.8))

                    # diff_t1 = np.abs(t1_out.data.cpu().numpy()[0, 0] - t1.data.cpu().numpy()[0, 0])
                    diff_t1 = t1_out.data.cpu().numpy()[0, 0] - t1.data.cpu().numpy()[0, 0]
                    diff_t1 = (trunc(diff_t1*255+135)).astype(np.uint8)
                    io.imsave(save_path + str(cnt) + '_t1_diff.png', diff_t1)


                elif args.modality == "t2":
                    if args.input_modality == "t1":
                        t2_out = network(t1_in)
                    elif args.input_modality == "t2":
                        t2_out = network(t2_in)

                        # t2_out, features = network(t2_in)

                        # for layer in range(len(features)):
                        #     feature = np.squeeze(np.mean(np.clip(features[layer].detach().cpu().numpy(), a_min=0, a_max=None), 1))
                        #     feature_img = (255*(feature - feature.min())/(feature.max() - feature.min())).astype(np.uint8)
                        #     scale = (240//feature_img.shape[0], 240//feature_img.shape[1])
                        #     feature_img = zoom(feature_img, scale, order=2)
                        #     io.imsave(feature_save_path + str(cnt) + '_layer' + str(layer+1) + '.png', feature_img)                            

                            # feature = np.clip(features[layer].detach().cpu().numpy(), a_min=0, a_max=None)[0,:,:,:]
                            # for c in range(feature.shape[0]):
                            #     feat = feature[c]
                            #     feat = feat[feat.shape[0]//4:-feat.shape[0]//4, feat.shape[0]//4:-feat.shape[0]//4]
                            #     feature_img = (255*(feat - feat.min())/(feat.max() - feat.min())).astype(np.uint8)
                            #     scale = (240//feature_img.shape[0], 240//feature_img.shape[1])
                            #     feature_img = zoom(feature_img, scale, order=2)
                            #     io.imsave(feature_save_path + str(cnt) + '_layer' + str(layer+1) + '_channel' + str(c+1) + '.png', feature_img)

                    if args.input_normalize == "mean_std":
                    ### 按照 x*std + mean把图像变回原来的特征范围
                        t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
                        t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]

                        # t1_img = ((t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean) * 255).astype(np.uint8)
                        # t2_img = ((t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean) * 255).astype(np.uint8)
                        # t1_out_img = ((t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean) * 255).astype(np.uint8)
                        # t2_out_img = ((t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean) * 255).astype(np.uint8)

                        t2_img = (np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                        t2_in_img = (np.clip(t2_in.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                        t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)

                        # print("t2_img range:", t2_img.max(), t2_img.min())
                        # print("t2_out_img range:", t2_out_img.max(), t2_out_img.min())

                    # if cnt > 1000:
                    #     save_path = snapshot_path + '/result_case_best/' + 'diff2/'
                    #     if not os.path.exists(save_path):
                    #         os.makedirs(save_path)                           
                        
                    #     t2_img = t2_img[40:200, 55:215]
                    #     t2_in_img = t2_in_img[40:200, 55:215]
                    #     t2_out_img = t2_out_img[40:200, 55:215]
                        
                    #     io.imsave(save_path + str(cnt) + '_t2_gt.png', bright(t2_img, 0, 0.8))
                    #     io.imsave(save_path + str(cnt) + '_t2_out.png', bright(t2_out_img, 0, 0.8))
                    #     io.imsave(save_path + str(cnt) + '_t2_in.png', bright(t2_in_img, 0, 0.8))

                    #     diff_t2 = (trunc(t2_img - t2_out_img +135)).astype(np.uint8)
                    #     io.imsave(save_path + str(cnt) + '_t2_diff.png', diff_t2)
                        
                    #     diff = np.abs(t2_img - t2_out_img)
                    #     fig = plt.figure()
                    #     plt.axis('off')
                    #     plt.imshow(diff, cmap='jet',vmin=0, vmax=30)
                    #     plt.savefig(save_path + str(cnt) + '_error_map.png', bbox_inches='tight',pad_inches = 0)


                # if t1_out is not None:                    
                #     t1_out[t1_out < 0.0] = 0.0
                #     # print("t1_recon range:", t1_out.max(), t1_out.min())
                #     t1_out_img = (t1_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                #     io.imsave(save_path + str(cnt) + '_t1_out.png', bright(t1_out_img,0,0.8))
                                    
                #     diff_t1 = t1_out.data.cpu().numpy()[0, 0] - t1.data.cpu().numpy()[0, 0]
                #     diff_t1 = (trunc(diff_t1*255 +135)).astype(np.uint8)
                #     io.imsave(save_path + str(cnt) + '_t1_diff.png', diff_t1)
                    

                # if t2_out is not None:
                #     t2_out[t2_out < 0.0] = 0.0
                #     # print("t2_recon range:", t2_out.max(), t2_out.min())
                #     t2_out_img = (t2_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                #     io.imsave(save_path + str(cnt) + '_t2_out.png', bright(t2_out_img,0,0.8))
                    
                #     diff_t2 = t2_out.data.cpu().numpy()[0, 0] - t2.data.cpu().numpy()[0, 0]
                #     diff_t2 = (trunc(diff_t2*255 +135)).astype(np.uint8)
                #     io.imsave(save_path + str(cnt) + '_t2_diff.png', diff_t2)         
                           

            # if t1_out is not None:
            #     t1_out_img[t1_out_img < 0.0] = 0.0
            #     t1_img[t1_img < 0.0] = 0.0
            #     MSE = mean_squared_error(t1_img, t1_out_img)
            #     PSNR = peak_signal_noise_ratio(t1_img, t1_out_img)
            #     SSIM = structural_similarity(t1_img, t1_out_img)
            #     t1_MSE_all.append(MSE)
            #     t1_PSNR_all.append(PSNR)
            #     t1_SSIM_all.append(SSIM)
            #     print("[t1 MRI] MSE:", MSE, "PSNR:", PSNR, "SSIM:", SSIM)


            if t2_out is not None:
                t2_out_img[t2_out_img < 0.0] = 0.0
                t2_img[t2_img < 0.0] = 0.0
                MSE = mean_squared_error(t2_img, t2_out_img)
                PSNR = peak_signal_noise_ratio(t2_img, t2_out_img)
                SSIM = structural_similarity(t2_img, t2_out_img)
                NMSE = nmse(t2.data.cpu().numpy()[0, 0], t2_out.data.cpu().numpy()[0, 0])
                t2_MSE_all.append(MSE)
                t2_PSNR_all.append(PSNR)
                t2_SSIM_all.append(SSIM)
                t2_NMSE_all.append(NMSE)
                print("[t2 MRI] MSE:", MSE, "PSNR:", PSNR, "SSIM:", SSIM, "NMSE:", NMSE)

            # if cnt > 20:
            #     break

        # print("[T1 MRI:] average MSE:", np.array(t1_MSE_all).mean(), "average PSNR:", np.array(t1_PSNR_all).mean(), "average SSIM:", np.array(t1_SSIM_all).mean())
        print("[T2 MRI:] average MSE:", np.array(t2_MSE_all).mean(), "average PSNR:", np.array(t2_PSNR_all).mean(), "average SSIM:", np.array(t2_SSIM_all).mean(), "average NMSE:", np.array(t2_NMSE_all).mean())
        print("[T2 MRI:] std MSE:", np.array(t2_MSE_all).std(), "std PSNR:", np.array(t2_PSNR_all).std(), "std SSIM:", np.array(t2_SSIM_all).std(), "std NMSE:", np.array(t2_NMSE_all).std())


    elif args.phase == "diff":
        save_mode_path = os.path.join(snapshot_path, 'iter_100000.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'])
        network.eval()
        cnt = 0
        save_path = snapshot_path + '/result_case_diff/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for sampled_batch in tqdm(testloader, ncols=70):
            print('processing ' + str(cnt) + ' image')
            ct_in, ct, mri_in, mri = sampled_batch['ct_in'].cuda(), sampled_batch['ct'].cuda(), \
                                     sampled_batch['mri_in'].cuda(), sampled_batch['mri'].cuda()

            for idx, lam in enumerate([0, 0.3, 0.5, 0.7, 1]):
                domainness = [torch.tensor(lam).cuda().float().reshape((1, 1))]
                with torch.no_grad():
                    fusion_out = network(ct_in, mri_in, domainness)[0][0]
                fusion_out[fusion_out < 0.0] = 0.0
                fusion_img = (fusion_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)

                diff_ct = fusion_out.data.cpu().numpy()[0, 0] - ct.data.cpu().numpy()[0, 0]
                diff_ct = (trunc(diff_ct*255 +135)).astype(np.uint8)

                diff_mri = fusion_out.data.cpu().numpy()[0, 0] - mri.data.cpu().numpy()[0, 0]
                diff_mri = (trunc(diff_mri*255 +135)).astype(np.uint8)

                io.imsave(save_path + 'diff_' + str(cnt) + '_'+ str(lam) + '_ct.png', diff_ct)
                io.imsave(save_path + 'diff_' + str(cnt) + '_'+ str(lam) + '_mri.png', diff_mri)

            cnt = cnt + 1
            if cnt > 3:
                break
