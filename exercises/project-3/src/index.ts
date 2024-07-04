import * as utils from './utils.ts';

const clearButton = document.getElementById('clear-button') as HTMLButtonElement;
const predictionParagraph = document.getElementById("prediction") as HTMLParagraphElement;

if (clearButton === null || predictionParagraph === null) {
  throw new Error("One of the elements not found.");
}

clearButton.onclick = () => {
  utils.resetCanvas();
  utils.clearRect();
  predictionParagraph.textContent = "";
}
