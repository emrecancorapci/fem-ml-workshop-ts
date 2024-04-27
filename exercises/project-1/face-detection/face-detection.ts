import '@tensorflow/tfjs';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as faceDetection from '@tensorflow-models/face-detection';
import { drawFaceBox, startWebcam, takePicture } from '../utils';

const init = async () => {
  const webcamButton = document.getElementById('webcam');
  const captureButton = document.getElementById('pause');
  const video = document.getElementById('video') as HTMLVideoElement;
  const model = await getModel();

  if (webcamButton) webcamButton.onclick = () => startWebcam(video);
  if (captureButton) captureButton.onclick = () => takePicture(video, predictWith(model));
};

const getModel = async () => {
  const detectionModel = await faceDetection.createDetector(
    faceDetection.SupportedModels.MediaPipeFaceDetector,
    {
      runtime: 'tfjs',
    },
  );

  return detectionModel;
};

const predictWith = (model: faceDetection.FaceDetector) => {
  const predict = async (img: HTMLCanvasElement) => {
    // Run the model on the image
    const predictions = await model.estimateFaces(img, {
      flipHorizontal: false,
    });
    // Show the results
    console.log('Predictions:', predictions);
    drawFaceBox(img, predictions);
  };

  return predict;
};

init();
