import tf from '@tensorflow/tfjs-node';
import fs from 'node:fs';
import path from 'node:path';

import {
  IMAGE_SIZE,
} from './config';

const __dirname = import.meta.dirname;

const trainImagesDir = path.join(__dirname, '../data/train');
const testImagesDir = path.join(__dirname, '../data/test');


export function getTrainData() {
  const data = loadImages(trainImagesDir)

  const imageSensors = data.map(({ imageTensors }) => imageTensors);
  const labels = data.map(({ labels }) => labels);

  return {
    images: tf.concat(imageSensors),
    labels: tf.oneHot(tf.tensor1d(labels, 'int32'), 2),
  };
}

export function getTestData() {
  const data = loadImages(testImagesDir);

  const imageSensors = data.map(({ imageTensors }) => imageTensors);
  const labels = data.map(({ labels }) => labels);

  return {
    images: tf.concat(imageSensors),
    labels: tf.oneHot(tf.tensor1d(labels, 'int32'), 2),
  };
}

function loadImages(dir: string) {
  return fs.readdirSync(dir).map((file) => generateTensorAndLabel(dir, file));
}

function generateTensorAndLabel (dir: string, file: string) {
  const filePath = path.join(dir, file);
  const buffer = fs.readFileSync(filePath);

  const imageTensors = tf.node
    .decodeImage(buffer)
    .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
    .expandDims();

  const isSquare = file.toLocaleLowerCase().endsWith('square.png');
  const isTriangle = file.toLocaleLowerCase().endsWith('triangle.png');

  const labels = isSquare ? 0 : isTriangle ? 1 : -1;

  return { imageTensors, labels };
}