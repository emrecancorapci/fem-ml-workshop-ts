import { model } from './create-model';
import { getTestData, getTrainData } from './get-data';

export async function trainModel() {
  const { images: trainImages, labels: trainLabels } = getTrainData();

  await model.fit(trainImages, trainLabels, {
    // Epochs are number of times or steps the model is trained on the dataset.
    // More epochs can lead to better results and longer train duration.
    // But it can also lead to overfitting. In other words model memorizes the
    // training data and cannot generalize to new data.
    epochs: 10,
    // Number of samples to use in one training step.
    batchSize: 5,
    // Validation split is the percentage of the dataset used for validation.
    // The model is not trained on this data. It is used to evaluate the model.
    validationSplit: 0.2,
  })

  const { images: testImages, labels: testLabels } = getTestData();

  const result = model.evaluate(testImages, testLabels);

  if (Array.isArray(result)) {
    const loss = result[0].dataSync()[0].toFixed(3);
    const accuracy = result[1].dataSync()[0].toFixed(3);

    console.log(`Loss: ${loss}, Accuracy: ${accuracy}`);
  } else {
    console.log(result.toString());
  }
}