import * as tmImage from "@teachablemachine/image";

/* 
Head tilt detection using a pre-trained model 

Detects head tilt using a pre-trained model. The model trained with the
Teachable Machine app. It detects user's head tilt left or right.
Model is probably too small to detect your head tilt accurately. So
you have to train a new model with your own data. With Teachable Machine
you can do that in a few minutes.

https://teachablemachine.withgoogle.com/
*/

type Prediction = {
  className: string;
  probability: number;
};

const model = await loadModel("../my_model");
const webcam = new tmImage.Webcam(640, 480, true);

(function init() {
  const startButton = document.getElementById("start") as HTMLButtonElement;

  if (!startButton) throw new Error("startButton not found");

  startButton.onclick = () => setupWebcam();
})();

async function setupWebcam() {
  await webcam.setup();
  await webcam.play();

  window.requestAnimationFrame(async () => loop());
  document.getElementById("webcam-container")?.appendChild(webcam.canvas);
}

async function loop() {
  webcam.update();
  await predict();
  window.requestAnimationFrame(async () => loop());
}

async function predict() {
  const predictions = await model.predict(webcam.canvas);
  const mostLikelyPrediction = (acc: Prediction, cur: Prediction) => {
    return cur.probability > acc.probability ? cur : acc;
  };
  const prediction = predictions.reduce(mostLikelyPrediction);

  console.log(prediction.className);
}

async function loadModel(from: string) {
  const modelPath = `${from}/model.json`;
  const metadataPath = `${from}/metadata.json`;

  const mlModel = await tmImage.load(modelPath, metadataPath);
  // const maxPredictions = mlModel.getTotalClasses();

  return mlModel;
}
