import { trainModel } from "./train-drawings";

// If you have this error:
// node:internal/modules/cjs/loader:1577
//   return process.dlopen(module, path.toNamespacedPath(filename));
//
// Do as in the link below
// https://github.com/tensorflow/tfjs/issues/8176#issuecomment-2089451054

await trainModel();