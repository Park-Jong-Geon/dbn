echo 0
taskset -c 48-95 python dbn_inputperturbation.py \
    --config config_dsb/c10_frnrelu_AtoABC_InputPerturbation.yaml \
    --save ./checkpoints/dbn/c10/inputperturbation0p7new/0 \
    --mixup_alpha 0.7
echo 1