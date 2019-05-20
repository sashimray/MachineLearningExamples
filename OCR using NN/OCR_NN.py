#OCR using Neural Network 
#Refer to : https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

character_mnist = keras.datasets.mnist ## dataset of handwritten characters from MNIST
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] ## specify class

def main():
    ## get the data set 
    (train_images, train_labels), (test_images, test_labels) = character_mnist.load_data()    
    
    ## Preprocess. scale the values to range of 0 to 1. 
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Setup Layers 
    ## Flatten() : transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
    ## First Dense Neural Layer: 128 nodes 
    ## Second Dense Neural layer: 10 node . This returns an array of 10 probability scores that sum to 1.
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    #Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
    #Optimizer —This is how the model is updated based on the data it sees and its loss function.
    #Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model using training set
    model.fit(train_images, train_labels, epochs=5)
    
    
    # Use the test set now and print accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    
    ## Predict outcome of test images. A prediction is an array of 10 numbers.
    predictions = model.predict(test_images)

    # show output for some sample test immages
    num_rows = 5
    num_cols = 5
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions, test_labels, test_images)
#      plt.subplot(num_rows, 2*num_cols, 2*i+2)
#      plot_value_array(i, predictions, test_labels)


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
if __name__== "__main__":
    main()