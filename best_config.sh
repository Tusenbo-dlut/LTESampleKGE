# Best Configuration for TransE
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/drugbank --model TransE --x_ops p.d.b -lte -n 1 -b 512 -d 600 -id 600 -od 600 -hd 0.1 -g 24.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --freq_based_subsampling -save models/TransE_xos_pdb_freq_drugbank --test_batch_size 32
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/BioSNAP --model TransE --x_ops p -lte -n 1 -b 512 -d 1000 -id 1000 -od 1000 -hd 0.1 -g 24.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --uniq_based_subsampling -save models/TransE_xos_p_uniq_drugbank --test_batch_size 32

# Best Configuration for DistMult
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/drugbank --model TransE --x_ops p.d.b -lte -n 1 -b 512 -d 600 -id 600 -od 600 -hd 0.1 -g 24.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --freq_based_subsampling -save models/TransE_xos_pdb_freq_drugbank --test_batch_size 32
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/BioSNAP --model TransE --x_ops p -lte -n 1 -b 512 -d 2000 -id 2000 -od 2000 -hd 0.1 -g 24.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --uniq_based_subsampling -save models/TransE_xos_p_uniq_drugbank --test_batch_size 32

# Best Configuration for ComplEx
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/drugbank --model TransE --x_ops p.d.b -lte -n 1 -b 512 -d 600 -id 1200 -od 1200 -hd 0.1 -g 6.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --freq_based_subsampling -save models/TransE_xos_pdb_freq_drugbank --test_batch_size 32
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/BioSNAP --model TransE --x_ops p -lte -n 1 -b 512 -d 1000 -id 2000 -od 2000 -hd 0.1 -g 6.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --uniq_based_subsampling -save models/TransE_xos_p_uniq_drugbank --test_batch_size 32

# Best Configuration for RotatE
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/drugbank --model TransE --x_ops p.d.b -lte -n 1 -b 512 -d 600 -id 1200 -od 1200 -hd 0.1 -g 24.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --freq_based_subsampling -save models/TransE_xos_pdb_freq_drugbank --test_batch_size 32
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda --do_test --data_path data/BioSNAP --model TransE --x_ops p -lte -n 1 -b 512 -d 1000 -id 2000 -od 2000 -hd 0.1 -g 24.0 -a 1.0 -adv -lr 0.01 --max_steps 20000 --uniq_based_subsampling -save models/TransE_xos_p_uniq_drugbank --test_batch_size 32