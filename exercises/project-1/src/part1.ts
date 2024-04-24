import "@tensorflow/tfjs";
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { handleFilePicker, showResult } from "./utils";

const init = async () => {
  // Load the model
  const model = await cocoSsd.load();
  // Get the predict function
  const predict = predictWith(model);
  // Get the input element
  handleFilePicker(predict);
}

const predictWith = (model: cocoSsd.ObjectDetection) => {
  const predict = async (img: HTMLImageElement) => {
    // Run the model on the image
    const predictions = await model.detect(img);
    // Show the results
    console.log('Predictions:', predictions);
    showResult(predictions);
  }

  return predict;
}

init();