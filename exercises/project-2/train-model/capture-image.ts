import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator';
import * as tf from '@tensorflow/tfjs-latest';

/**
 * Processes an image and returns a tensor of the processed image
 * @param image image to be processed
 */

export default async function captureImage(webcam: WebcamIterator) {
  const image = await webcam.capture();
  const processedImage = tf.tidy(function imageProcessFx() {
    return image.expandDims(0).toFloat().div(127).sub(1).resizeBilinear([224, 224]);
  });
  
  image.dispose();

  return processedImage;
}
