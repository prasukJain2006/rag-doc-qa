# Neural Networks

## Perceptron and Activation Functions
A perceptron takes weighted inputs, adds a bias, and passes the result through an activation function. ReLU outputs zero for negative inputs and the input itself for positive inputs. Sigmoid squashes values into the range 0 to 1. Tanh squashes values into the range minus 1 to 1. ReLU is the common default for hidden layers because it helps avoid vanishing gradients.

## Backpropagation and Loss
Backpropagation computes gradients of the loss with respect to each weight using the chain rule, and the optimiser updates the weights to reduce the loss. For classification, cross-entropy loss is standard. For regression, mean squared error is standard. Training proceeds in epochs, where one epoch is a full pass over the training data.

## Optimisers
Adam is a widely used optimiser that adapts the learning rate for each parameter. A common starting learning rate for Adam is 0.001. Too high a learning rate can make training unstable, while too low a learning rate makes training very slow. Batch size controls how many examples are used for one gradient update.

## Regularisation with Dropout
Dropout randomly disables a fraction of neurons during training, for example a dropout rate of 0.5. This prevents neurons from co-adapting too much and reduces overfitting. Dropout is turned off at test time, so predictions use the full network.

## Convolutional Neural Networks
Convolutional Neural Networks, or CNNs, are used for images. Convolutional layers learn local patterns such as edges and textures. Pooling layers reduce the spatial size and make the model more robust to small shifts. A CNN for image classification usually ends with fully connected layers and a softmax output.

## Recurrent Networks and LSTM
Recurrent Neural Networks process sequences one step at a time and keep a hidden state. Plain RNNs suffer from the vanishing gradient problem on long sequences. LSTM, or Long Short-Term Memory, adds gates that control what to remember and what to forget, which helps it learn long-range dependencies in text and time series.
