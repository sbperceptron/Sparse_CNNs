# srun --nodes=3-3 --ntasks=3 --cpus-per-task=3 ...
srun --nodes=1-36 --ntasks=1 --cpus-per-task=36 python main_train.py -f 3 -w 6 -b 36 -i ../data > my1.log 2>&1 &
echo $! > save_pid1.txt
srun --nodes=1-36 --ntasks=1 --cpus-per-task=36 python main_train.py -f 3 -w 12 -b 36 -i ../data > my2.log 2>&1 &
echo $! > save_pid2.txt
srun --nodes=1-36 --ntasks=1 --cpus-per-task=36 python main_train.py -f 3 -w 24 -b 36 -i ../data > my3.log 2>&1 &
echo $! > save_pid3.txt
srun --nodes=1-36 --ntasks=1 --cpus-per-task=36 python main_train.py -f 6 -w 6 -b 36 -i ../data > my4.log 2>&1 &
echo $! > save_pid4.txt
srun --nodes=1-36 --ntasks=1 --cpus-per-task=36 python main_train.py -f 6 -w 12 -b 36 -i ../data > my5.log 2>&1 &
echo $! > save_pid5.txt
srun --nodes=1-36 --ntasks=1 --cpus-per-task=36 python main_train.py -f 6 -w 24 -b 36 -i ../data > my6.log 2>&1 &
echo $! > save_pid6.txt

