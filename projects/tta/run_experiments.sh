# # ensemble experiment
# NUM_ENSEMBLES=5
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="ensemble_${NUM_ENSEMBLES}mdls_gn_10wrmup_3nratio_loco"
#         # --model_name "resnet18" \
# for CENTER in "CRCEO" "UVA" "JH" # "PCC" "PMCC" "CRCEO" "UVA" "JH"  
# do
#     python ensemble_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --batch_size 32 \
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --split_test False \
#         --concat_test_train False \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0 \
#         --warmup_epochs 10
# done 


# # sngp experiment
# INSTANCE_NORM=True
# GROUP="sngp_inst-nrm_loco"
# LR=0.001
# WEIGHT_DECAY=0.0001
# for CENTER in "UVA" # "CRCEO" "JH" "PCC" "PMCC" 
# do
#     python sngp_experiment.py \
#         --name "${GROUP}_${CENTER}_32bz_lre-3" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --lr $LR \
#         --weight_decay $WEIGHT_DECAY \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --batch_size 32 \
#         --instance_norm $INSTANCE_NORM                
# done 


# # baseline experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# # GROUP="baseline_gn_avgprob_3ratio_loco"
# # GROUP="baseline_gn_2x2pz_3ratio_loco"
# GROUP="baseline_bn_10wrmup_3ratio_loco-splttst"
# # GROUP="baseline_gn_avgprob_3ratio_1poly_loco"
# # GROUP="sam_baseline_gn_e-4rho_loco"
# # GROUP="baseline_bn_inst-nrm_loco"
#         # --slurm_qos "deadline" \
#         # --slurm_account "deadline" \
#         # --slurm_exclude "gpu034,gpu017" \
# for CENTER in  "PCC"  "CRCEO"  "PMCC" "JH" # "UVA"
# do
#     python baseline_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --slurm_gres "gpu:a40:1" \
#         --cluster "slurm" \
#         --batch_size 32 \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --split_test True \
#         --concat_test_train False \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0 \
#         --use_poly1_loss False \
#         --eps 1.0 \
#         --needle_mask_threshold 0.6 \
#         --patch_size_mm 5.0 5.0 \
#         --strides 1.0 1.0 \
#         --warmup_epochs 10 \
#         --lr 0.0001
# done



# # baseline experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="baseline_gn_3nratio"
# for fold in 0 1 2 3 4
# do
#     python baseline_experiment.py \
#         --name "${GROUP}_${fold}" \
#         --group "${GROUP}" \
#         --slurm_gres "gpu:a40:1" \
#         --cluster "slurm" \
#         --cohort_selection_config "kfold" \
#         --fold $fold \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0 \
#         --use_poly1_loss False \
#         --eps 1.0 \
#         --patch_size_mm 5.0 5.0 \
#         --strides 1.0 1.0 \
#         --lr 0.0001
# done


# # ttt experiment
# QUERY_PATCH=True
# SUPPORT_PATCHES=2
# GROUP="ttt_${SUPPORT_PATCHES}+1sprt_0.1beta_5e-4adptlr_2nratio_loco"

# for CENTER in  "JH" "PCC" "PMCC" "CRCEO" "UVA" 
# do
#     python ttt_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --slurm_exclude "gpu034,gpu017" \
#         --cohort_selection_config "loco" \
#         --benign_to_cancer_ratio_train 2.0 \
#         --leave_out $CENTER \
#         --include_query_patch $QUERY_PATCH \
#         --num_support_patches $SUPPORT_PATCHES \
#         --epochs 50 \
#         --joint_training False \
#         --avg_core_probs_first True \
#         --adaptation_steps 1 \
#         --adaptation_lr 0.0005 \
#         --beta_byol 0.1
# done 


# # mt3 experiment
# QUERY_PATCH=True
# SUPPORT_PATCHES=2
# GROUP="mt3_${SUPPORT_PATCHES}+1sprt_0.1beta_5e-4innlr_2nratio_loco"

# for CENTER in  "JH" "PCC" # "PMCC" "UVA" "CRCEO" # 
# do
#     python mt3_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --slurm_exclude "gpu034,gpu017" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --include_query_patch $QUERY_PATCH \
#         --num_support_patches $SUPPORT_PATCHES \
#         --benign_to_cancer_ratio_train 2.0 \
#         --epochs 50 \
#         --inner_steps 1 \
#         --inner_lr 0.0005 \
#         --beta_byol 0.1
# done 


# # vicreg pretrain experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# # GROUP="vicreg_pretrn_1024zdim_gn_300ep_3ratio_loco"
# GROUP="vicreg_pretrn_1024zdim_bn_150ep_2ratio_loco"
# for CENTER in "PCC" #"PMCC" "UVA" "CRCEO" # "JH"   
# do
#     python vicreg_pretrain_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --slurm_exclude "gpu034,gpu017" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 2.1 \
#         --epochs 150 \
#         --proj_output_dim 1024 \
#         --cov_coeff 1.0 \
#         --linear_lr 0.001 \
#         --linear_epochs 15
# done


# # vicreg pretrain experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# GROUP="vicreg_pretrn_512zdim_1e-3lr_10linprob_200ep_1ratio_bn_f"
# for FOLD in 0 # 1 2 3 4
# do
#     python vicreg_pretrain_experiment.py \
#         --name "${GROUP}_${FOLD}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --cohort_selection_config "kfold" \
#         --fold $FOLD \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 1.0 \
#         --proj_output_dim 512 \
#         --cov_coeff 1.0 \
#         --epochs 200 \
#         --linear_lr 0.001 \
#         --linear_epochs 10
# done


# # vicreg finetune core experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# # GROUP="vicreg_1024-300finetune_1e-4lr_8heads_64qk128v_8corebz_gn_loco"
# # GROUP="vicreg_finetune_1e-4backlr_1e-4headlr_8heads_transformer_gn_loco_batch10_newrunep"
# # GROUP="vicreg_1024-150corefintun_1ratio_5e-4lr_5e-5bck_64qk128v_8corebz_bn_3ratio_bz8_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_loco"
# # checkpoint_path_name="vicreg_pretrn_2048zdim_gn_300ep_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_3ratio_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_bn_150ep_3ratio_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_bn_150ep_1ratio_loco"

# GROUP="TRUSformer_1024-300corefintun_3ratio_5e-4lr_5e-5bck_64qk128v_8corebz_bn_3ratio_bz8_loco"
# checkpoint_path_name="vicreg_pretrn_1024zdim_1e-3lr_10linprob_300ep_3ratio_bn_f"
# for CENTER in "JH" #"PCC" "PMCC" "UVA" "CRCEO" 
# do
#     python core_finetune_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --slurm_exclude "gpu034,gpu017" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0 \
#         --epochs 50 \
#         --core_batch_size 8 \
#         --nhead 8 \
#         --qk_dim 64 \
#         --v_dim 128 \
#         --checkpoint_path_name $checkpoint_path_name \
#         --backbone_lr 0.00005 \
#         --head_lr 0.0005 \
#         --batch_size 1 \
#         --dropout 0.0 \
#         # --prostate_mask_threshold -1
# done


# vicreg finetune core experiment
INSTANCE_NORM=False
USE_BATCH_NORM=True
GROUP="TRUSformer_nopos_1024-300corefintun_3ratio_5e-4lr_5e-5bck_64qk128v_8corebz_bn_3ratio_bz8_loco"
checkpoint_path_name="vicreg_pretrn_1024zdim_1e-3lr_10linprob_300ep_3ratio_bn_f"
for FOLD in 0 
do
    python core_finetune_experiment.py \
        --name "${GROUP}_${FOLD}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --cohort_selection_config "kfold" \
        --fold $FOLD \
        --instance_norm $INSTANCE_NORM \
        --use_batch_norm $USE_BATCH_NORM \
        --benign_to_cancer_ratio_train 3.0 \
        --epochs 50 \
        --batch_size 16 \
        --backbone_lr 0.0005 \
        --head_lr 0.0005 \
        --nhead 8 \
        --nlayer 12 \
        --checkpoint_path_name $checkpoint_path_name \
        --prostate_mask_threshold -1
done



# # vicreg finetune experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# # GROUP="vicreg_1024-300finetune_1e-3lr_gn_loco"
# # GROUP="vicreg_1024-300finetune_1e-4lr_avgprob_gn_crtd3ratio_loco2"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_3ratio_loco"
# # GROUP="vicreg_1024-150finetune_1e-4lr_avgprob_bn_3ratio_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_bn_150ep_3ratio_loco"
# GROUP="vicreg_1024-150finetune_1e-4lr_bn_3ratio_loco"
# checkpoint_path_name="vicreg_pretrn_1024zdim_bn_150ep_3ratio_loco"
# for CENTER in  "JH" "PCC" "PMCC" "UVA" "CRCEO"  # 
# do
#     python finetune_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --slurm_exclude "gpu034,gpu017" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --benign_to_cancer_ratio_train 3.0 \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --epochs 50 \
#         --train_backbone True \
#         --checkpoint_path_name $checkpoint_path_name \
#         --backbone_lr 0.0001 \
#         --head_lr 0.0001
# done


# # Divemble experiment
# NUM_ENSEMBLES=5
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="ensemble-shrd-fe_gn_${NUM_ENSEMBLES}mdls_3ratio_loco"
# # GROUP="Divemble-logt_gn_${NUM_ENSEMBLES}mdls_0.5var0.05cov_3ratio_loco"
# # GROUP="Divemble-shrd_gn_${NUM_ENSEMBLES}mdls_0var0.5cov_3ratio_loco"
# # GROUP="Divemble_gn_${NUM_ENSEMBLES}mdls_crctd_loco"
# for CENTER in  "PCC" "PMCC" "UVA" "CRCEO" # "JH" 
# do
#     python divemble_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --epochs 50 \
#         --benign_to_cancer_ratio_train 3.0 \
#         --var_reg 0.0 \
#         --cov_reg 0.0
# done


# # MI ensemble experiment
# NUM_ENSEMBLES=5
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="MIensemble_10mi_${NUM_ENSEMBLES}mdls_gn_10wrmup_3ratio_loco"
# # GROUP="MIensemble_.1mi_.05var.01cov_${NUM_ENSEMBLES}mdls_3ratio_gn_loco"
#         # --slurm_qos "deadline" \
#         # --slurm_account "deadline" \
# for CENTER in "JH" "PCC" "PMCC" "CRCEO" "UVA" #  
# do
#     python MI_ensemble_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --epochs 50 \
#         --benign_to_cancer_ratio_train 3.0 \
#         --var_coeff 0.0 \
#         --cov_coeff 0.0 \
#         --mi_coeff 10.0
# done