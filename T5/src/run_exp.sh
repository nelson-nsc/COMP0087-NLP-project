TRAIN_OPTION=$1
BACKBONE="t5-base"

python finetune_t5.py --backbone_model $BACKBONE --train_option $TRAIN_OPTION --n_gpu 1

for i in 0 1 2 3 4 5 6 7 8 9
do
    echo inference_t5.py --backbone_model $BACKBONE --train_option $TRAIN_OPTION --repeat $i
done