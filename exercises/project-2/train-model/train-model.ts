import * as tf from '@tensorflow/tfjs-latest';
import * as tfd from '@tensorflow/tfjs-data';
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator';

const recordButtons = document.getElementsByClassName(
  'record-button',
) as HTMLCollectionOf<HTMLButtonElement>;
const buttonsContainer = document.getElementById('buttons-container') as HTMLDivElement;

const trainButton = document.getElementById('train') as HTMLButtonElement;
const predictButton = document.getElementById('predict') as HTMLButtonElement;
const statusElement = document.getElementById('status') as HTMLSpanElement;

let webcam: WebcamIterator,
  initialModel: tf.LayersModel,
  mouseDown: boolean,
  modelOutput: tf.Sequential,
  exampleTensors: tf.Tensor<tf.Rank>,
  labelTensors: tf.Tensor<tf.Rank>;

const totals = [0, 0];

const options = {
  labels: ['left', 'right'],
  learningRate: 0.001,
  batchSizeFraction: 0.4,
  epochs: 30, // Number of times to train the model less is faster but less accurate, more is slower but more accurate
  denseUnits: 100, // Number of units in the dense layer
};

let states = {
  isTraining: false,
  isPredicting: false,
};

buttonsContainer.onmousedown = async function handleAddExample(event) {
  const labelIndex = Number(event.target !== recordButtons[0]);
  mouseDown = true;
  console.log(labelIndex);
  const total = document.getElementById(`${options.labels[labelIndex]}-total`) as HTMLSpanElement;

  while (mouseDown) {
    addExample(labelIndex);
    total.innerText = String(++totals[labelIndex]);

    await tf.nextFrame();
  }
};

buttonsContainer.onmouseup = () => (mouseDown = false);

trainButton.onclick = async () => {
  train();
  statusElement.innerText = 'Training...';
  statusElement.style.display = 'block';
};

/**
 * 1. Access the webcam
 * 2. Load the MobileNet model
 * 3. Update the ui
 */

async function init() {
  const webcamElement = document.getElementById('webcam') as HTMLVideoElement;
  if (!webcamElement) throw new Error('Webcam element not found');
  webcamElement.width = 224;
  webcamElement.height = 224;
  webcam = await tfd.webcam(webcamElement);

  initialModel = await loadModel().then((model) => {
    statusElement.innerText = 'Model loaded';
    return model;
  });

  const controller = document.getElementById('controller') as HTMLDivElement;
  if (!controller) throw new Error('Controller not found');

  controller.style.display = 'block';
}

/**
 * Captures an image from a webcam, processes it. Uses a machine learning model to make a prediction based on the image. Then converts a given label into a tensor format and stores the example and label tensors.
 * 
 * If there are already example tensors, it appends the new example and label tensors to the existing ones and disposes of the previous ones to manage memory efficiently. 
 */

async function addExample(labelIndex: number) {
  const image = await webcam.capture();
  const processedImage = await processImage(image);
  const example = initialModel.predict(processedImage);
  // tf.oneHot is turning the label into a one-hot vector. Instead of using right or left, we'll use the index of the label
  // tf.tensor1d is turning the label into a tensor of size 1. This is needed to use tf.oneHot
  const labelTensor = tf.tidy(() =>
    tf.oneHot(tf.tensor1d([labelIndex]).toInt(), options.labels.length),
  );

  if (exampleTensors == undefined) {
    if (Array.isArray(example)) throw new Error('example variable is an array');

    exampleTensors = tf.keep(example);
    labelTensors = tf.keep(labelTensor);
  } else {
    const previousExampleTensors = exampleTensors;
    exampleTensors = tf.keep(previousExampleTensors.concat(example, 0));
    const previousY = labelTensors;
    labelTensors = tf.keep(previousY.concat(example, 0));

    previousExampleTensors.dispose();
    previousY.dispose();
  }

  labelTensor.dispose();
}

async function processImage(image: tf.Tensor3D) {
  const processedImage = tf.tidy(function imageProcessFx() {return image.expandDims(0).toFloat().div(127).sub(1)});
  image.dispose();

  return processedImage;
}

/* 
To create a model:
  1. Choose type of algorithm (sequential, functional, or graph)
  2. Add layers
  3. Compile model with optimizer
  4. Split data by batches
  5. Train model
*/

function train() {
  states.isTraining = true;
  if (!exampleTensors) throw new Error('There is no examples to train on. Add some examples first');

  modelOutput = tf.sequential({
    layers: [
      tf.layers.flatten({
        inputShape: initialModel.outputs[0].shape.slice(1),
      }),
      tf.layers.dense({
        units: options.denseUnits,
        activation: 'relu', // Rectified Linear Unit
        kernelInitializer: 'varianceScaling',
        useBias: true,
      }),
      tf.layers.dense({
        units: options.labels.length,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling',
        useBias: true,
      }),
    ],
  });

  // There are more options for the optimizer. See the docs
  // for more info https://www.tensorflow.org/js/guide/optimizers
  const optimizer = tf.train.adam(options.learningRate);

  modelOutput.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const batchSize = Math.floor(exampleTensors.shape[0] * options.batchSizeFraction);

  modelOutput.fit(exampleTensors, labelTensors, {
    batchSize,
    epochs: options.epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        statusElement.innerText = `Batch ${batch} of ${exampleTensors.shape[0]} complete. Loss: ${logs?.loss.toFixed(4)} Accuracy: ${logs?.acc.toFixed(4)}`;
      },
    },
  });

  states.isTraining = false;
}

/**
 * Loads the MobileNet model. Extracts the single layer that is used for classification
 */
async function loadModel(): Promise<tf.LayersModel> {
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
  );
  const layer = mobilenet.getLayer('conv_pw_13_relu');

  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

init();
