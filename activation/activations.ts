// Activation functions

import {Matrix, MatrixOps} from "../ops/matrix.ops";

export class Activations {
    static relu(x: number): number {
        return Math.max(0, x);
    }

    static gelu(x: number): number {
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    }

    static applyActivation(matrix: Matrix, activation: (x: number) => number): Matrix {
        const result = MatrixOps.create(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                result.data[i][j] = activation(matrix.data[i][j]);
            }
        }
        return result;
    }
}
