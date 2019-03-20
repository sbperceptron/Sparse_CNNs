# Model training
Load the model parameters, input data (the full point cloud data for hard negative mining and the crop data for training) and intialized weights.

Then supply the crops traindata to the batch generator function. Which will iterate over the train data in steps equal to the batch size and feed the data from all the input data.

The batch data is provided to the nn_model2 function which uses each sample input and get a prediction score after passing through the stacked model. Then we use the l1 hinge loss function to caliculate the loss by comparing the prediction with the label information. Then the loss value is used to caliculate the gradients for all the weights. The gradients obtained for all the data samples in the batch are averaged and multiplied with the learning rate to update the weights. Along with the learning rate we use stochastic gradient descent and the L2 weight decay parameters to update the weights of the model

In this way the weights are updated for every batch and every epoch. The weights obtained for every epoch are stored in the directory specified.

Additionally, each time the model completes 10 epochs we perform the hard negative mining. In this process, the full point cloud without any positive data is supplied to the model and the loss values are obtained. The locations corresponding to the top 10 loss values are cropped and added to the negative training set for every full point cloud in the input.

   