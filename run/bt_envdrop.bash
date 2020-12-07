name=entity-graph-vln-stage2

flag="--train auglistener
      --selfTrain
      --test_only 0
      --batchSize 32

      --aug data/aug_paths.json
      --speaker snap/envdrop-speaker/state_dict/best_val_unseen_bleu
      --load snap/entity-graph-vln-stage1/state_dict/best_val_unseen

      --test_obj 0
      --features places365
      --featdropout 0.4

      --accumulateGrad
      --optim rms
      --lr 1e-5
      --iters 300000
      --maxAction 35"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python r2r_src/train.py $flag --name $name
