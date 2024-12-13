nohup python train_SOTA.py --root_path ../../MRI/BRATS_100patients/image_100patients_4X/ --max_iterations 10\
    --gpu 0 --model_name mmr_mamba --batch_size 1 --base_lr 0.001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
    --exp S6_shallow4C_Ampv2+05_mul20%Res_lr001_4X_240 > "logs/$(date +"%Y%m%d-%H%M")_S6_shallow4C_Ampv2+05_mul20%Res_lr001_4X_240.log" 2>&1 &


# nohup python train_knee.py --root_path kneeS8_shallow4C_Ampv2+05_2CNNsum5%2Res_lr001_4X../../MRI/BRATS_100patients/image_100patients_8X/ --max_iterations 100000 \
#     --gpu 1 --model_name mmr_mamba_knee --batch_size 4 --base_lr 0.001 --MRIDOWN 8X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal True --modality t2 --input_normalize mean_std \
#     --exp kneeS8_shallow4C_AmpSumv2+05_2CNNmul10%2Res_lr001_8X > "logs/$(date +"%Y%m%d-%H%M")_kneeS8_shallow4C_AmpSumv2+05_2CNNmul10%2Res_lr001_8X.log" 2>&1 &
