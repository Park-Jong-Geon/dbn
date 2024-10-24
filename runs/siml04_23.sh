echo 0
python dsb_renewal_fat.py \
    --config config_dsb/c10_frnrelu_AtoshABCnew.yaml \
    --feature_name feature.layer2stride1 \
    --tag AtoABC \
    --fat 3 \
    --optim_bs 128 \
    --joint 2 \
    --T 5 \
    --joint_depth 6 \
    --version v1.1.4 \
    --forget 3 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --linear_noise \
    --mixup_alpha 0.4
echo 1