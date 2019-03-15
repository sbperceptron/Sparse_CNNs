import multiprocessing
import time
from multiprocessing import Pool
from random import randint
def f(x):
  time.sleep(randint(10,100)/1000)
  return x*x

def parallel(r,c):
  pool2=Pool(processes=8)
  for i in pool2.imap_unordered(f, range(r),c):
    print(i)
  pool2.close()
  pool2.join()


parallel(20,2)
