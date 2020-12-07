name=envdrop-speaker

flag="--train speaker
      --test_only 0
      --batchSize 64

      --features places365
      --dropout 0.6

      --optim adam
      --lr 1e-4
      --iters 80000
      --maxAction 35"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python r2r_src/train.py $flag --name $name
