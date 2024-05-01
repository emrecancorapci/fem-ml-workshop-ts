let clickX = new Array();
let clickY = new Array();
let clickDrag = new Array();
let paint: boolean; // check this

const labels = ["circle", "triangle"];
const canvas = document.getElementsByTagName("canvas")[0] as HTMLCanvasElement | undefined;
const link = document.getElementById("download-link") as HTMLAnchorElement | undefined;

if (!canvas) throw new Error("Canvas not found");
if (!link) throw new Error("Link not found");

const context = canvas.getContext("2d");
if (!context) throw new Error("Canvas not found");

export function displayPrediction(label: number) {
  let prediction = labels[label];

  var predictionParagraph = document.getElementsByClassName("prediction")[0];
  predictionParagraph.textContent = prediction;
}

export function clearRect(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  context.clearRect(0, 0, canvas.width, canvas.height);
}

export function getCanvas() {
  return canvas;
}

export function resetCanvas() {
  clickX = new Array();
  clickY = new Array();
  clickDrag = new Array();
  paint;
}

canvas.addEventListener("mousedown", function startDrawing(e) {
  paint = true;
  addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
  redraw(context, canvas);
});

canvas.addEventListener("mousemove", function onDrawing(e) {
  if (paint) {
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
    redraw(context, canvas);
  }
});

canvas.addEventListener("mouseup", function enDrawing() {
  paint = false;
});

link.addEventListener(
  "click",
  function () {
    link.href = canvas.toDataURL();
    link.download = "drawing.png";
  },
  false
);

function addClick(x: number, y: number, dragging: boolean = false) {
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

function redraw(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas

  context.strokeStyle = "#000000";
  context.lineJoin = "round";
  context.lineWidth = 5;
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);

  for (var i = 0; i < clickX.length; i++) {
    context.beginPath();
    if (clickDrag[i] && i) {
      context.moveTo(clickX[i - 1], clickY[i - 1]);
    } else {
      context.moveTo(clickX[i] - 1, clickY[i]);
    }
    context.lineTo(clickX[i], clickY[i]);
    context.closePath();
    context.stroke();
  }
}

