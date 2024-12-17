## node 1
accelerate launch --mixed_precision no --num_machines=1 --num_processes=1 --machine_rank=0 \
--main_process_ip=10.31.22.186 --main_process_port=24000 mnist_accelerate.py --epochs=10

## node 2
accelerate launch --multi_gpu --mixed_precision no --num_machines=2 --num_processes=2 --machine_rank=0 \
--main_process_ip=10.31.22.186 --main_process_port=24000 mnist_accelerate.py --epochs=10

## node 3
accelerate launch --multi_gpu --mixed_precision no --num_machines=3 --num_processes=3 --machine_rank=0 \
--main_process_ip=10.31.22.186 --main_process_port=24000 mnist_accelerate.py --epochs=10

## node 4
accelerate launch --multi_gpu --mixed_precision no --num_machines=4 --num_processes=4 --machine_rank=0 \
--main_process_ip=10.31.22.186 --main_process_port=24000 mnist_accelerate.py --epochs=10
