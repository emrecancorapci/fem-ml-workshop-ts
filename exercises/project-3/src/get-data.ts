import * as tf from '@tensorflow/tfjs-node-gpu';
import fs from 'node:fs';
import path from 'node:path';

const trainImagesDir = path.join(__dirname, '../data/train');
const testImagesDir = path.join(__dirname, '../data/test');

function loadImages(dir: string) {
  const imageAndLabels = fs.readdirSync(dir).map(function (file) {
    const filePath = path.join(dir, file);
    const buffer = fs.readFileSync(filePath);

    const imageTensors = tf.node.decodeImage(buffer).resizeNearestNeighbor([20, 20]).expandDims();
    
    const isSquare = file.toLocaleLowerCase().endsWith('square.png');
    const isTriangle = file.toLocaleLowerCase().endsWith('triangle.png');

    const labels = isSquare ? 0 : isTriangle ? 1 : -1;

    return { imageTensors, labels };
  });

  return imageAndLabels;
  
}

export const loadTrainData = () => loadImages(trainImagesDir);
export const loadTestData = () => loadImages(testImagesDir);

export function getTrainData() {
  const data = loadTrainData();

  const imageSensors = data.map(({ imageTensors }) => imageTensors);
  const labels = data.map(({ labels }) => labels);

  return {
    images: tf.concat(imageSensors),
    labels: tf.oneHot(tf.tensor1d(labels, 'float32'), 0)
  }
}

export function getTestData() {
  const data = loadTestData();

  const imageSensors = data.map(({ imageTensors }) => imageTensors);
  const labels = data.map(({ labels }) => labels);

  return {
    images: tf.concat(imageSensors),
    labels: tf.oneHot(tf.tensor1d(labels, 'float32'), 0)
  }
}
