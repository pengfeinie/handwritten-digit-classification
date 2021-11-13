### 3.2 Convolutional Neural Network for Deep Learning

http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

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
