import '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { showResult, startWebcam, takePicture } from './utils';

const init = async () => {
  const webcamButton = document.getElementById('webcam');
  const captureButton = document.getElementById('pause');
  const video = document.getElementById('video') as HTMLVideoElement;
  const model = await cocoSsd.load();

  if (webcamButton) webcamButton.onclick = () => startWebcam(video);
  if (captureButton) captureButton.onclick = () => takePicture(video, predictWith(model));
};
const predictWith = (model: cocoSsd.ObjectDetection) => {
  const predict = async (img: HTMLCanvasElement) => {
    // Run the model on the image
    const predictions = await model.detect(img);
    // Show the results
    console.log('Predictions:', predictions);
    showResult(predictions);
  };

  return predict;
};

init();
