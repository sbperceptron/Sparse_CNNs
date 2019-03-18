The building blocks of the network building
Activations.py
1. Activation functions are used after every layer 
present in the model
2. The Activation function we are using here is the 
RELU activation. 
3. Which is defined as max(0,x)
batch_geneartor.py
The Batch generator function main goal is to constantly provide a batch of data to the network.
1. The batch generator takes in all the input pointclouds and the labels
2. Then for each of the pointcloud and label we are
going to extract the feature vectors
3. And the extracted feature vector and the label are
stored into a list
4. when we have collected batchsize number of samples 
we pass the collected data to the network
feature_extractor.py
Feature extractor takes in the raw point cloud data and
extract the feature vectors
1. This is done by first converting the raw point cloud data into a grid discretized using 0.2,0.2,0.2 along the x,y,z axis. Every cell in the grid encapsulates a set of points in it.
2. From this grid representation we find the feature vector representation
3. This is done for each and every grid cell in the grid. firstly,we take all the points falling into a cell and we compute the covariance matrix of all the points falling into the cell. Then for the covariance matrix we find the eigen values.
4. The eigen values obtained are used to caliculate the shape factors
5. There are totally three shape factors(Linear,planar and shperical shape factors) which we use as three features for every cell in the grid. 
6. Along with the shape factors we use the mean of intensity values and variance of intensity values falling in the grid cell as two other features for a cell in the grid.
7. Finally we add a binary value a the sixth shape factor. Which denote whether the cell is occupied or not
8. We repeat this process for all the cells in the grid and return a dictionary containing the Feature vector location and the Features so that we can directly operate on this data instead of dense array.
loss_functions.py
uses a l1 hinge loss function to caliculate the loss based on the model prediction and ground truth
max(0,1-ypred.ytrue)
readcalib.py
read the calib information from the calibration data provided with the kitti dataset
readlabels.py
Reads the label information from the label data and the calib data provided with the kitti dataset
HNM.py
The hard negative mining is performed in this file
1. For each fullpointcloud data we remove the points corresponding to the object of interest(Pedestrians for example) and then we convert the full point cloud into grid and then eventually into feature vector representation using the feature extraction tools
2. Then from the feature vectors extracted, locations for the full pc we take feature vectors from a window(equal to the receptive field size of the object) around each location and give the model to predict.
3. since we have removed all the objects from the full pc an ideal model should recognize all of them as negative samples. But we can see the model predicting some of this negative samples as positives. From this false positives we pick the top 10 high confidence predictions and add them to our training set.
4. This process is repeated for all the pointclouds in the input.
Sparse_CNN_2layer_forward.py
The Sparse_CNN_2layer_forward model takes in the feature vector dictionary, weights and biases initialized or the updated weights as the input
and does the sparse convolution on the input and returns the convolution output
Sparse_CNN_2layer_backward.py
This function caliculates the gradients that are required for backpropagation to correct the model weights and biases.