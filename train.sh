file="./main.py"
args="--cuda \
    --seed 9 \
    --dbscale \
    --data_aug \
    --save_json \
    --save_model \
    --save_multiple \
    --epochs 500 \
    --lrate 55e-5 \
    --batch_size 8 \
    --num_workers 16 \
    --data_dir ./data \
    --save_dir ./results \
    --model_file baseline.pth"

python "$file" $args
