vote3deep master

the input folder should look like
main
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

Currently there are four hyper parameters to tune,
  * hard negative mining batch size
  * feature vector threshold
  * window threshold
  * resolution