# Deep Neural network(DNN) Trained Model for MNIST

## The python class "Net":
represents the structure of the DNN in its __init__() method. The structure contains commonly occurring layers in it's 3 blocks:
1. 3x3 Convolution Layers - This is where the kernels are defined and are able to extract features
2. Batch_normalization Layers - Makes the feature representation of the Kernels stronger by normalizaing them
3. Drop-out Layers - Randomly drop out neurons with the probability of 0.15 
4. Max-Pool Layers - Used to double receptive field and reduce the no. of params. Takes the maximum value in a 2x2 window and jumps with a stride of 2
5. 1x1 Convolution (AntMan) Layers - Used to reduce the no. of channels to reduce the no. of params.
6. Average Pooling Layer - the output layer. It takes the average of the 3x3 input and there is one value for each class.
7. Log_softmax Layer - The Likelihoods for each Class should all add up to 1. This layer takes care of that. 

## forward Method:
used to define the forward pass network. After every Conv blocks in Block 1 and 2, we pass it through relu, batch norm and drop-outs
We also take the max-pool at the end of each block and reduce the channels using "antman"

## Summary/ Cuda Block:

We first check whether Cuda is available. If Cuda is available we set the device as Cuda. We also create a network Object from the Net class and save it in GPU.
The summary shows that the no. of parameters used is 19972.

## Dataset Download and Load Block:
Download the Train and Test data -> Transform the Images to Tensors -> 
Normalize the Images by subtracting the mean and dividing the standard deviation from all the images.

We also create iterators(train_loader and test_loader) which go over the images in batches (according to the batch_size). 
These iterators are used by the training function to train the model in batches


## train function:
1. batches are saved one by one into GPU and fed to the model. Based on the predictions, Negative Log Likelihood loss (NLL) it calculated. 

2. NLL gets sad when the classes are misclassified and says that the loss is high. For correct classification, NLL is happy and says the loss is low. 

3. The total loss for all the images in a batch combined needs to be minimized by the network.

4. Once the loss is calculated, gradients are propagated backwards (back propagation) to the layers using "loss.backward()"

5. Based on the gradients, the kernel weights are modified using optimizer.step() function.


## test function:

1. While testing we dont need to train and back_propagate. We just need to evaluate the result for each batch. 

2. We calculate the no. of correct predictions to determine the overall acccuracy.


## Epoch Block

1. All of the Batches need to be fed multiple times to the Network for better results. Each such cycle is calles the epoch. 
Here we use 19 epochs to get an accuracy > 99.4 

2. We gradually decrease the learning rate from 0.1 -> 0.01 using a ReduceOnPlateau scheduler, 
which checks whether the model is stuck in the current learning rate and if it is, it reduces it by a factor of 10.


















