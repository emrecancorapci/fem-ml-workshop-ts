import tf from '@tensorflow/tfjs-node';
import { DENSE_LAYER_UNITS, FILTERS, IMAGE_SIZE, LEARNING_RATE } from './config';

// https://cs231n.github.io/convolutional-networks/

/**
 * Kernel size is the size of the filter matrix for the convolution.
 * Kernel size of 3 means a `3x3` filter matrix.
 */
const kernelSize = [3, 3];

/**
 * Input shape is the shape of the input image.
 * In this case, it is a `28x28` image with `4` channels.
 * 
 * The `4` channels are the `RGBA` values of the image.
 */
const inputShape = [IMAGE_SIZE, IMAGE_SIZE, 4];


/**
 * Number of classes to classify the input image.
 * 
 * If this number is different than the number of classes in the dataset,
 * the model will not be able to classify the input image.
 */
const numberOfClasses = 2;

const optimizer = tf.train.adam(LEARNING_RATE);

const layers = [
    // Convolutional layer is used to detect features in the input image.
    // This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    tf.layers.conv2d({
        inputShape,
        filters: FILTERS,
        kernelSize,
        // There are also two different activation layers that Sigmoid and Softmax.
        // Softmax usually used in last layer of the model.
        // Because this is the first layer of the model, we use ReLU.
        activation: 'relu',
    }),

    // Pooling layer is used to reduce the spatial dimensions of the output volume.
    // It is used to reduce the amount of parameters and computation in the network.
    tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }),

    // Flatten layer is used to flatten the input, so that it can be fed to the next layer.
    // It flattens each batch in its inputs to 1D (making the output 2D).
    tf.layers.flatten(),

    // Dense layer is a regular densely-connected NN layer.
    // It is used to classify the features extracted by the convolutional layers.
    // More units in the dense layer means more features to classify the input image.
    tf.layers.dense({
        units: DENSE_LAYER_UNITS,
        activation: 'relu',
    }),

    // Last layer of the model.
    // It is used to classify the input image.
    // Units is the number of classes to classify the input image.
    tf.layers.dense({
        units: numberOfClasses,
        activation: 'softmax',
    })
]


const model = tf.sequential({
    layers
});

model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

export { model };