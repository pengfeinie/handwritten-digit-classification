* [1\. Development Environment](#1-development-environment)
* [2\. MNIST Handwritten Digit Classification Dataset](#2-mnist-handwritten-digit-classification-dataset)
    * [2\.1 Loading Dataset](#21-loading-dataset)
    * [2\.2 Prepare Pixel Data](#22-prepare-pixel-data)
* [3\. Handwritten Digit Classification](#3-handwritten-digit-classification)
  * [3\.1 Fully Connected Neural Network for Machine Learning](#31-fully-connected-neural-network-for-machine-learning)
    * [3\.1\.1 Loading Dataset](#311-loading-dataset)
    * [3\.1\.2 Prepare Pixel Data](#312-prepare-pixel-data)
    * [3\.1\.3 Define Model](#313-define-model)
    * [3\.1\.4 Evaluate Model](#314-evaluate-model)
    * [3\.1\.5 Present Results](#315-present-results)
    * [3\.1\.6 Complete Example](#316-complete-example)
  * [3\.2 Convolutional Neural Network for Deep Learning](#32-convolutional-neural-network-for-deep-learning)
    * [3\.2\.1 Loading Dataset](#321-loading-dataset)
    * [3\.2\.2 Prepare Pixel Data](#322-prepare-pixel-data)
    * [3\.2\.3 Define Model](#323-define-model)
    * [3\.2\.4 Evaluate Model](#324-evaluate-model)
    * [3\.2\.5 Present Results](#325-present-results)
    * [3\.2\.6 Complete Example](#326-complete-example)

In this tutorial, we’ll give you a step by step walk-through of how to build a hand-written digit classifier using the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. For someone new to deep learning, this exercise is arguably the “Hello World” equivalent. Although each step will be thoroughly explained in this tutorial, it will certainly benefit someone who already has some theoretical knowledge of the working of CNN. Also, some knowledge of [TensorFlow](https://www.tensorflow.org/) is also good to have, but not necessary.

## 1. Development Environment

This tutorial assumes that you are using standalone Keras running on top of TensorFlow with Python 3.

| Python 3.8 | Tensorflow 2.6 |
| ---------- | -------------- |

## 2. MNIST Handwritten Digit Classification Dataset

The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.  Although the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use convolutional deep learning neural networks for image classification from scratch. This includes how to develop a robust test harness for estimating the performance of the model, how to explore improvements to the model, and how to save the model and later load it to make predictions on new data.

MNIST is a widely used dataset for the hand-written digit classification task. It consists of 70,000 labeled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). The task at hand is to train a model using the 60,000 training images and subsequently test its classification accuracy on the 10,000 test images.

The dataset that is being used here is the *[MNIST digits classification dataset](https://keras.io/api/datasets/mnist/)*. Keras is a deep learning API written in Python and MNIST is a dataset provided by this API. This dataset consists of 60,000 training images and 10,000 testing images. It is a decent dataset for individuals who need to have a go at pattern recognition as we will perform in just a minute!

When the Keras API is called, there are four values returned namely- *x_train, y_train, x_test, and y_test*. Do not worry, I will walk you through this.

#### 2.1 Loading Dataset

We know some things about the dataset.

For example, we know that the images are all pre-aligned (e.g. each image only contains a hand-drawn digit), that the images all have the same square size of 28×28 pixels, and that the images are grayscale.

Therefore, we can load the images and reshape the data arrays to have a single color channel.

We also know that there are 10 classes and that classes are represented as unique integers.

We can, therefore, use a one hot encoding for the class element of each sample, transforming the integer into a 10 element binary vector with a 1 for the index of the class value, and 0 values for all other classes. We can achieve this with the *to_categorical()* utility function.

The *load_dataset()* function implements these behaviors and can be used to load the dataset.

```python
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
```

#### 2.2 Prepare Pixel Data

We know that the pixel values for each image in the dataset are unsigned integers in the range between black and white, or 0 and 255.

We do not know the best way to scale the pixel values for modeling, but we know that some scaling will be required.

A good starting point is to [normalize the pixel values](https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/) of grayscale images, e.g. rescale them to the range [0,1]. This involves first converting the data type from unsigned integers to floats, then dividing the pixel values by the maximum value.

The *prep_pixels()* function below implements these behaviors and is provided with the pixel values for both the train and test datasets that will need to be scaled.

```python
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm
```

## 3. Handwritten Digit Classification

### 3.1 Fully Connected Neural Network for Machine Learning 

To define the architecture of this first fully connected neural network, we'll once again use the Keras API and define the model using the [`Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential) class. Note how we first use a [`Flatten`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) layer, which flattens the input so that it can be fed into the model. 

In this next block, you'll define the fully connected layers of this simple work.

We'll first build a simple neural network consisting of two fully connected layers and apply this to the digit classification task. Our network will ultimately output a probability distribution over the 10 digit classes (0-9). This first architecture we will be building is depicted below:

![](https://pengfeinie.github.io/images/mnist_2layers_arch.png)

#### 3.1.1 Loading Dataset

[loading-dataset](https://github.com/pengfeinie/handwritten-digit-classification#21-loading-dataset)

#### 3.1.2 Prepare Pixel Data

[prepare-pixel-data](https://github.com/pengfeinie/handwritten-digit-classification#22-prepare-pixel-data)

#### 3.1.3 Define Model

Before training the model, we need to define a few more settings. 

* *Loss function* — This defines how we measure how accurate the model is during training. As was covered in lecture, during training we want to minimize this function, which will "steer" the model in the right direction.
* *Optimizer* — This defines how the model is updated based on the data it sees and its loss function.
* *Metrics* — Here we can define metrics used to monitor the training and testing steps. In this example, we'll look at the *accuracy*, the fraction of the images that are correctly classified.

We'll start out by using a stochastic gradient descent (SGD) optimizer initialized with a learning rate of 0.1. Since we are performing a categorical classification task, we'll want to use the [cross entropy loss](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_crossentropy).

You'll want to experiment with both the choice of optimizer and learning rate and evaluate how these affect the accuracy of the trained model. 

```python
# define cnn model
def define_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

#### 3.1.4 Evaluate Model

We're now ready to train our model, which will involve feeding the training data (`train_images` and `train_labels`) into the model, and then asking it to learn the associations between images and labels. We'll also need to define the batch size and the number of epochs, or iterations over the MNIST dataset, to use during training. 

In Lab 1, we saw how we can use `GradientTape` to optimize losses and train models with stochastic gradient descent. After defining the model settings in the `compile` step, we can also accomplish training by calling the [`fit`](https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential#fit) method on an instance of the `Model` class. We will use this to train our fully connected model.

Now that we've trained the model, we can ask it to make predictions about a test set that it hasn't seen before. In this example, the `test_images` array comprises our test dataset. To evaluate accuracy, we can check to see if the model's predictions match the labels from the `test_labels` array. Use the [`evaluate`](https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential#evaluate) method to evaluate the model on the test dataset!

```python
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kFold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kFold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], 
        dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, 
                            epochs=10, batch_size=32, 
                            validation_data=(testX, testY), verbose=0)
        # evaluate model
        acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories
```

#### 3.1.5 Present Results

Once the model has been evaluated, we can present the results.

There are two key aspects to present: the diagnostics of the learning behavior of the model during training and the estimation of the model performance. These can be implemented using separate functions.

First, the diagnostics involve creating a line plot showing model performance on the train and test set during each fold of the k-fold cross-validation. These plots are valuable for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset.

We will create a single figure with two subplots, one for loss and one for accuracy. Blue lines will indicate model performance on the training dataset and orange lines will indicate performance on the hold out test dataset. The *summarize_diagnostics()* function below creates and shows this plot given the collected training histories.

```python
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()
```

Next, the classification accuracy scores collected during each fold can be summarized by calculating the mean and standard deviation. This provides an estimate of the average expected performance of the model trained on this dataset, with an estimate of the average variance in the mean. We will also summarize the distribution of scores by creating and showing a box and whisker plot.

The *summarize_performance()* function below implements this for a given list of scores collected during model evaluation.

```python
# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()
```

#### 3.1.6 Complete Example

We need a function that will drive the test harness.

This involves calling all of the define functions.

```python
# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)
```

We now have everything we need; Running the example prints the classification accuracy for each fold of the cross-validation process. This is helpful to get an idea that the model evaluation is progressing.

![image-20211103172931937](https://pengfeinie.github.io/images/image-20211103172931937.png)

Next, a diagnostic plot is shown, giving insight into the learning behavior of the model across each fold.

In this case, we can see that the model generally achieves a good fit, with train and test learning curves converging. There is no obvious sign of over- or underfitting.

![image-20211103172818144](https://pengfeinie.github.io/images/image-20211103172818144.png)

Finally, a box and whisker plot is created to summarize the distribution of accuracy scores.

![image-20211103172856976](https://pengfeinie.github.io/images/image-20211103172856976.png)

### 3.2 Convolutional Neural Network for Deep Learning

For those of you new to this concept, CNN is a deep learning technique to classify the input automatically (well, after you provide the right data). Over the years, CNN has found a good grip over classifying images for computer visions and now it is being used in healthcare domains too. This indicates that CNN is a reliable deep learning algorithm for an automated end-to-end prediction. CNN essentially extracts ‘useful’ features from the given input automatically making it super easy for us!

![end to end process of CNN](https://pengfeinie.github.io/images/42220New.jpg)

We will define a simple convolutional neural network with 2 convolution layers followed by two fully connected layers. Below is the model architecture we will be using for our CNN. We follow up each convolution layer with RelU activation function and a max-pool layer. RelU introduces non-linearity and max-pooling helps with removing noise.

![](https://pengfeinie.github.io/images/2021-10-31_124519.png)

#### 3.2.1 Loading Dataset

[loading-dataset](https://github.com/pengfeinie/handwritten-digit-classification#21-loading-dataset)

#### 3.2.2 Prepare Pixel Data

[prepare-pixel-data](https://github.com/pengfeinie/handwritten-digit-classification#22-prepare-pixel-data)

#### 3.2.3 Define Model

Next, we need to define a baseline convolutional neural network model for the problem.

The model has two main aspects: the feature extraction front end comprised of convolutional and pooling layers, and the classifier backend that will make a prediction.

For the convolutional front-end, we can start with a single [convolutional layer](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) with a small filter size (3,3) and a modest number of filters (32) followed by a [max pooling layer](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/). The filter maps can then be flattened to provide features to the classifier.

Given that the problem is a multi-class classification task, we know that we will require an output layer with 10 nodes in order to predict the probability distribution of an image belonging to each of the 10 classes. This will also require the use of a softmax activation function. Between the feature extractor and the output layer, we can add a dense layer to interpret the features, in this case with 100 nodes.

All layers will use the [ReLU activation function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) and the He weight initialization scheme, both best practices.

We will use a conservative configuration for the stochastic gradient descent optimizer with a [learning rate](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/) of 0.01 and a momentum of 0.9. The [categorical cross-entropy](https://machinelearningmastery.com/cross-entropy-for-machine-learning/) loss function will be optimized, suitable for multi-class classification, and we will monitor the classification accuracy metric, which is appropriate given we have the same number of examples in each of the 10 classes.

The *define_model()* function below will define and return this model.

```python
# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), 
                     strides=(1, 1), 
                     padding="same", 
                     kernel_size=(3, 3), 
                     filters=32,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(strides=(1, 1), 
                     padding="same", 
                     kernel_size=(3, 3), 
                     filters=64,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(3136, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

#### 3.2.4 Evaluate Model

After the model is defined, we need to evaluate it.

The model will be evaluated using [five-fold cross-validation](https://machinelearningmastery.com/k-fold-cross-validation/). The value of *k=5* was chosen to provide a baseline for both repeated evaluation and to not be so large as to require a long running time. Each test set will be 20% of the training dataset, or about 12,000 examples, close to the size of the actual test set for this problem.

The training dataset is shuffled prior to being split, and the sample shuffling is performed each time, so that any model we evaluate will have the same train and test datasets in each fold, providing an apples-to-apples comparison between models.

We will train the baseline model for a modest 10 training epochs with a default batch size of 32 examples. The test set for each fold will be used to evaluate the model both during each epoch of the training run, so that we can later create learning curves, and at the end of the run, so that we can estimate the performance of the model. As such, we will keep track of the resulting history from each run, as well as the classification accuracy of the fold.

The *evaluate_model()* function below implements these behaviors, taking the training dataset as arguments and returning a list of accuracy scores and training histories that can be later summarized.

```python
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kFold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kFold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], 
        dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, 
                            epochs=10, batch_size=32, 
                            validation_data=(testX, testY), verbose=0)
        # evaluate model
        acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories
```

#### 3.2.5 Present Results

Once the model has been evaluated, we can present the results.

There are two key aspects to present: the diagnostics of the learning behavior of the model during training and the estimation of the model performance. These can be implemented using separate functions.

First, the diagnostics involve creating a line plot showing model performance on the train and test set during each fold of the k-fold cross-validation. These plots are valuable for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset.

We will create a single figure with two subplots, one for loss and one for accuracy. Blue lines will indicate model performance on the training dataset and orange lines will indicate performance on the hold out test dataset. The *summarize_diagnostics()* function below creates and shows this plot given the collected training histories.

```python
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()
```

Next, the classification accuracy scores collected during each fold can be summarized by calculating the mean and standard deviation. This provides an estimate of the average expected performance of the model trained on this dataset, with an estimate of the average variance in the mean. We will also summarize the distribution of scores by creating and showing a box and whisker plot.

The *summarize_performance()* function below implements this for a given list of scores collected during model evaluation.

```python
# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()
```

#### 3.2.6 Complete Example

We need a function that will drive the test harness.

This involves calling all of the define functions.

```python
# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)
```

We now have everything we need; Running the example prints the classification accuracy for each fold of the cross-validation process. This is helpful to get an idea that the model evaluation is progressing.

![image-20211102092040889](https://pengfeinie.github.io/images/image-20211102092040889.png)

Next, a diagnostic plot is shown, giving insight into the learning behavior of the model across each fold.

In this case, we can see that the model generally achieves a good fit, with train and test learning curves converging. There is no obvious sign of over- or underfitting.

![image-20211102092239587](https://pengfeinie.github.io/images/image-20211102092239587.png)

Finally, a box and whisker plot is created to summarize the distribution of accuracy scores.

![image-20211102092329134](https://pengfeinie.github.io/images/image-20211102092329134.png)



[classification-of-handwritten-digits-using-cnn/](https://www.analyticsvidhya.com/blog/2021/07/classification-of-handwritten-digits-using-cnn/)

[mnist-handwritten-digit-classification/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)

[introtodeeplearning](https://colab.research.google.com/github/aamini/introtodeeplearning/blob/master/lab2/Part1_MNIST.ipynb#scrollTo=_J72Yt1o_fY7)
https://matthewearl.github.io/2016/05/06/cnn-anpr/
