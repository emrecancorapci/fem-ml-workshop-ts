import('@tensorflow/tfjs').then(() => console.log('TensorFlow.js loaded'));
import { load as loadModel, ObjectDetection } from '@tensorflow-models/coco-ssd';
import { handleFilePicker, showResult } from '../utils';

const init = async () => {
  // Load the model
  const model = await loadModel();
  // Get the predict function
  const predict = predictWith(model);
  // Get the input element
  handleFilePicker(predict);
};
  
const predictWith = (model: ObjectDetection) => {
  const predict = async (img: HTMLImageElement) => {
    // Run the model on the image
    const predictions = await model.detect(img);
    // Show the results
    console.log('Predictions:', predictions);
    showResult(predictions);
  };

  return predict;
};

init();
