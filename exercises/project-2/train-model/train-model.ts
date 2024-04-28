import * as tf from '@tensorflow/tfjs-latest';
import * as tfd from '@tensorflow/tfjs-data';
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator';

import captureImage from './capture-image';

const recordButtons = document.getElementsByClassName(
  'record-button',
) as HTMLCollectionOf<HTMLButtonElement>;
const buttonsContainer = document.getElementById('buttons-container') as HTMLDivElement;

const trainButton = document.getElementById('train') as HTMLButtonElement;
const predictButton = document.getElementById('predict') as HTMLButtonElement;
const statusElement = document.getElementById('status') as HTMLSpanElement;

if (!buttonsContainer) throw new Error('Buttons container not found');
if (!trainButton) throw new Error('Train button not found');
if (!predictButton) throw new Error('Predict button not found');
if (!statusElement) throw new Error('Status element not found');

let webcam: WebcamIterator,
  initialModel: tf.LayersModel,
  createdModel: tf.LayersModel,
  exampleTensors: tf.Tensor<tf.Rank>,
  labelTensors: tf.Tensor<tf.Rank>,
  mouseDown: boolean;

const totals = [0, 0];
const labels = ['left', 'right'];

const options = {
  learningRate: 0.001,
  batchSizeFraction: 0.4,
  epochs: 30, // Number of times to train the model less is faster but less accurate, more is slower but more accurate
  denseUnits: 100, // Number of units in the dense layer
};

let isPredicting = false;

/**
 * 1. Access the webcam
 * 2. Load the MobileNet model
 * 3. Update the ui
 */
async function init() {
  const webcamElement = document.getElementById('webcam') as HTMLVideoElement;
  const controllerElement = document.getElementById('controller') as HTMLDivElement;

  if (!webcamElement) throw new Error('Webcam element not found');
  if (!controllerElement) throw new Error('Controller element not found');
  if (!statusElement) throw new Error('Status element not found');

  webcam = await tfd.webcam(webcamElement);

  initialModel = await loadModel().then((model) => {
    statusElement.style.display = 'none';
    controllerElement.style.display = 'block';

    return model;
  });
}

/**
 * Loads the MobileNet model. Extracts the single layer that is used for classification
 */
async function loadModel() {
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
  );

  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

/**
 * Handles the mouse down event. Adds an example to the example tensors and label tensors
 */
async function handleAddExample(labelIndex: number) {
  const total = document.getElementById(`${labels[labelIndex]}-total`);

  if (!total) throw new Error('Total element not found');

  while (mouseDown) {
    addExample(labelIndex);
    total.innerText = String(++totals[labelIndex]);

    await tf.nextFrame();
  }
}

/**
 * Captures an image from a webcam, processes it. Uses a machine learning model to make a prediction based on the image. Then converts a given label into a tensor format and stores the example and label tensors.
 *
 * If there are already example tensors, it appends the new example and label tensors to the existing ones and disposes of the previous ones to manage memory efficiently.
 *
 * `tf.oneHot` is turning the label into a one-hot vector. Instead of using right or left, we'll use the index of the label
 *
 * `tf.tensor1d` is turning the label into a tensor of size 1. This is needed to use `tf.oneHot`
 */
async function addExample(labelIndex: number) {
  const image = await captureImage(webcam);
  const example = initialModel.predict(image);

  const newLabelTensor = tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), labels.length));

  if (exampleTensors == undefined) {
    if (Array.isArray(example)) throw new Error('example variable is an array');
    exampleTensors = tf.keep(example);
    labelTensors = tf.keep(newLabelTensor);
  } else {
    const previousExampleTensors = exampleTensors;
    const previousLabelTensors = labelTensors;

    exampleTensors = tf.keep(previousExampleTensors.concat(example, 0));
    labelTensors = tf.keep(previousLabelTensors.concat(newLabelTensor, 0));

    previousExampleTensors.dispose();
    previousLabelTensors.dispose();
    newLabelTensor.dispose();
    image.dispose();
  }
}

buttonsContainer.onmousedown = async (event: MouseEvent) => {
  mouseDown = true;
  let labelIndex;
  for (labelIndex = 0; labelIndex < recordButtons.length; labelIndex++) {
    if (event.target === recordButtons[labelIndex]) break;
  }
  handleAddExample(labelIndex);
};

buttonsContainer.onmouseup = () => {
  mouseDown = false;
};

/* 
To create a model:
  1. Choose type of algorithm (sequential, functional, or graph)
  2. Add layers
  3. Compile model with optimizer
  4. Split data by batches
  5. Train model
*/
async function train() {
  if (!exampleTensors) throw new Error('You forgot to add examples before training');

  createdModel = tf.sequential({
    layers: [
      tf.layers.flatten({
        inputShape: initialModel.outputs[0].shape.slice(1),
      }),
      tf.layers.dense({
        units: options.denseUnits,
        activation: 'relu', // Rectified Linear Unit (ReLU)
        kernelInitializer: 'varianceScaling',
        useBias: true,
      }),
      tf.layers.dense({
        units: labels.length,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling',
        useBias: true,
      }),
    ],
  });

  const optimizer = tf.train.adam(options.learningRate);
  createdModel.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

  const batchSize = Math.floor(exampleTensors.shape[0] * options.batchSizeFraction);

  await createdModel.fit(exampleTensors, labelTensors, {
    batchSize,
    epochs: options.epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        if (!logs) {
          statusElement.innerHTML = 'Logs not found';
          return;
        }

        statusElement.innerHTML = 'Loss: ' + logs.loss.toFixed(5);
      },
    },
  });
}

trainButton.onclick = async () => {
  train();
  statusElement.style.display = 'block';
  statusElement.innerHTML = 'Training...';
};


/**
 * Makes a prediction on the image captured from the webcam. Uses the created model to make the prediction.
 */
async function predict() {
  const image = await captureImage(webcam);

  const initialModelPrediction = initialModel.predict(image);
  const predictions = createdModel.predict(initialModelPrediction);

  if (Array.isArray(predictions)) throw new Error('predictions is an array');

  const predictedClass = predictions.as1D().argMax();
  const classId = (await predictedClass.data())[0];

  console.log(labels[classId]);

  image.dispose();
  await tf.nextFrame();
}

predictButton.onclick = async () => {
  isPredicting = true;
  while (isPredicting) {
    predict();
  }
};

init();
