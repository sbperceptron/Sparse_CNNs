# vote3deep master
The latest version of the vote3deep cpu implementation. 
# Running the code
python main_train.py -h (for all the arguments requried for the algo to run)
run ./bash_script.sh (for running a batch of jobs)
## the input folder should look like
### main
  * kittisplit  
    * train  
	  * bin  
	  * labels  
	  * calibs  
    * test  
	  * bin  
	  * labels  
	  * calibs  

  * crops  
    * train  
      * positive  
        * Pedestrian  
	  * negative  
        * Pedestrian  
    * test  
      * positive  
        * Pedestrian  
      * split  
        * positive  
          * Pedestrian  
            * Easy  
            * Medium  
            * Hard  
        * negative  
          * Pedestrian  
            * Easy  
            * Medium  
            * Hard  

Currently there are three parameters to tune,
  
  * feature vector threshold
  * window threshold
  * resolution
