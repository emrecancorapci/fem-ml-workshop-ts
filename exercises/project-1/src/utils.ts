// Part 1
// -----------

import { DetectedObject } from "@tensorflow-models/coco-ssd";

export const showResult = (classes: DetectedObject[]) => {
  const predictionsElement = document.getElementById("predictions");
  const probsContainer = document.createElement("div");
  for (let i = 0; i < classes.length; i++) {
    probsContainer.innerText = `Prediction: ${classes[i].class}, Probability: ${classes[i].score}`;
  }

  if (predictionsElement) predictionsElement.appendChild(probsContainer);
};

export const IMAGE_SIZE = 512;

export const handleFilePicker = (callback: (img: HTMLImageElement) => Promise<void>) => {
  // Get the file input element
  const fileElement = document.getElementById("file");
  if (!fileElement) return;

  fileElement.addEventListener("change", (evt) => {
    const files = (evt.target as HTMLInputElement).files;
    if (!files) return;

    const file = files[0];

    if (!file.type.match("image.*")) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = document.createElement("img");
      if (typeof e.target?.result === 'string') {
        img.src = e.target.result;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
      }

      const loadedImgElement = document.getElementById("loaded-image");
      if (loadedImgElement) {
        loadedImgElement.appendChild(img);
        // loadedimg = img;
      }

      img.onload = async () => await callback(img);

      // img.onload = () => predict(img);
    };
    reader.readAsDataURL(file);
  });
};

// Part 2
// -----------

export const startWebcam = async (video: HTMLVideoElement) => {
  try {
    const stream = await navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: { width: 320, height: 185 },
      });
    video.srcObject = stream;
    // track = stream.getTracks()[0];
    video.onloadedmetadata = () => video.play();
  } catch (err) { }
};

// export const startWebcam = (video: HTMLVideoElement) => {
//   return navigator.mediaDevices
//     .getUserMedia({
//       audio: false,
//       video: { width: 320, height: 185 },
//     })
//     .then((stream) => {
//       video.srcObject = stream;
//       // track = stream.getTracks()[0];
//       video.onloadedmetadata = () => video.play();
//     })
//     .catch((err) => {
//       /* handle the error */
//     });
// };

export const takePicture = (video: HTMLVideoElement, callback: (img: HTMLCanvasElement) => void) => {
  const predictButton = document.getElementById("predict") as HTMLButtonElement;
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  if(!canvas) return;
  // const width = 320; // We will scale the photo width to this
  // const height = 185;
  const width = IMAGE_SIZE; // We will scale the photo width to this
  const height = IMAGE_SIZE;

  const context = canvas.getContext("2d");
  if(!context) return;

  canvas.width = width;
  canvas.height = height;
  context.drawImage(video, 0, 0, width, height);

  const outputEl = document.getElementById("predictions");
  if (!outputEl) return;
  // outputEl.appendChild(photo);
  outputEl.appendChild(canvas);

  predictButton.disabled = false;

  predictButton.onclick = async () => {
    await callback(canvas);
  };
};

// Part 3
// -----------

export interface Face {
  box: FaceBox;
  landmarks: FaceLandmark[];
}

export interface FaceBox {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
}

export interface FaceLandmark {
  x: number;
  y: number;
}

export const drawFaceBox = (photo: HTMLCanvasElement, faces: Face[]) => {
  // Draw box around the face detected ⬇️
  // ------------------------------------
  const faceCanvas = document.createElement("canvas");
  faceCanvas.width = IMAGE_SIZE;
  faceCanvas.height = IMAGE_SIZE;
  faceCanvas.style.position = "absolute";
  faceCanvas.style.left = `${photo.offsetLeft}px`;
  faceCanvas.style.top = `${photo.offsetTop}px`;
  const ctx = faceCanvas.getContext("2d");
  if(!ctx) return;

  ctx.beginPath();
  ctx.strokeStyle = "red";
  ctx.strokeRect(
    faces[0].box.xMin,
    faces[0].box.yMin,
    faces[0].box.width,
    faces[0].box.height
  );

  const webcamSection = document.getElementById("webcam-section");
  webcamSection?.appendChild(faceCanvas);
};
