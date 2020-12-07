name=entity-graph-vln
# entiry-graph-vln-stage1

flag="--train validlistener
      --test_only 0
      --batchSize 32

      --load snap/entity-graph-vln/state_dict/best_val_unseen
      --submit 0

      --test_obj 0
      --features places365
      --featdropout 0.3
      --mlWeight 0.20
      --dropout 0.5

      --feedback sample
      --optim rms
      --lr 1e-4
      --iters 80000
      --maxAction 35"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python r2r_src/train.py $flag --name $name
