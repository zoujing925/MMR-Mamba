# ## without guidance modality, compare the image domain performance with under_mri and kspace_recon as input.
# nohup python test_SOTA.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X/ \
#     --gpu 2 --model_name mambaIR --MRIDOWN 4X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp mambaIR2662_lr0.001_4X > "logs/test_$(date +"%Y%m%d-%H%M")_mambaIR2662_lr0.001_4X.log" 2>&1 & 


# nohup python test_SOTA.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X/ \
#     --gpu 1 --model_name panmamba_new --MRIDOWN 4X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp S6_shallow4C_Ampv2+05_mul10%Res_lr001_4X_240 > "logs/test_$(date +"%Y%m%d-%H%M")_S6_shallow4C_Ampv2+05_mul10%Res_lr001_4X_240.log" 2>&1 &

# nohup python test_SOTA.py --root_path ../../MRI/BRATS_100patients/image_100patients_8X/ \
#     --gpu 2 --model_name panmamba_baseline --MRIDOWN 8X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp S6_baseline_D1CNN1_lr001_8X_240 > "logs/test_$(date +"%Y%m%d-%H%M")_S6_baseline_D1CNN1_lr001_8X_240" 2>&1 &



# nohup python test_knee.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X_224/ \
#     --gpu 2 --model_name panmamba_knee --MRIDOWN 8X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp kneeS8_shallow4C_AmpSumv2+05_2CNNmul10%2Res_lr001_8X > "logs/test_$(date +"%Y%m%d-%H%M")_kneeS8_shallow4C_AmpSumv2+05_2CNNmul10%2Res_lr001_8X.log" 2>&1 &


# nohup python test_knee.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X_224/ \
#     --gpu 0 --model_name mambaIR --MRIDOWN 8X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp knee_mambaIR_lr001_8X > "logs/test_$(date +"%Y%m%d-%H%M")_knee_mambaIR_lr001_8X.log" 2>&1 &

# nohup python test_knee.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X_224/ \
#     --gpu 2 --model_name panmamba_origin --MRIDOWN 8X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp kneeS6SwapD4_lr001_4X > "logs/test_$(date +"%Y%m%d-%H%M")_kneeS6SwapD4_lr001_4X.log" 2>&1 &


# python test_knee.py --root_path /data/qic99/MRI_recon/fastMRI/ \
#     --gpu 0 --model_name DCAMSR --batch_size 4 --base_lr 0.0001 \
#     --exp DCAMSR_8x --model_name DCAMSR --phase test > "logs/test_$(date +"%Y%m%d-%H%M")_8X_DCAMSR.log" 2>&1 &

# nohup python test_SOTA.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X/ \
#     --gpu 3 --model_name winet --MRIDOWN 4X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp winet32_3hffftTanhCAresFres_lr001_4X_240 > \
#     "logs_cvpr25/test_$(date +"%Y%m%d-%H%M")_winet32_3hffftTanhCAresFres_lr001_4X_240.log" 2>&1 &


nohup python test_knee.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X_224/ \
    --gpu 3 --model_name winet_knee --MRIDOWN 8X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
    --exp knee_winet64_3FFTRI_wFilt1c1INres_HFfftTanhCAresTanhFres_add_lr001_8X > \
    "logs_cvpr25/test_$(date +"%Y%m%d-%H%M")_knee_winet64_3FFTRI_wFilt1c1INres_HFfftTanhCAresTanhFres_add_lr001_8X.log" 2>&1 &
