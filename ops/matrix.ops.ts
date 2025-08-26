import {Activations} from "../activation/activations";

export interface Matrix {
    data: number[][];
    rows: number;
    cols: number;
}

// Utility functions for matrix operations
export class MatrixOps {
    static create(rows: number, cols: number, initValue = 0): Matrix {
        return {
            data: Array(rows).fill(null).map(() => Array(cols).fill(initValue)),
            rows,
            cols
        };
    }

    static random(rows: number, cols: number, scale = 0.1): Matrix {
        const matrix = this.create(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                matrix.data[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return matrix;
    }

    static multiply(a: Matrix, b: Matrix): Matrix {
        if (a.cols !== b.rows) {
            throw new Error(`Cannot multiply matrices: ${a.cols} !== ${b.rows}`);
        }

        const result = this.create(a.rows, b.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    static add(a: Matrix, b: Matrix): Matrix {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Matrix dimensions must match for addition');
        }

        const result = this.create(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }

    static transpose(matrix: Matrix): Matrix {
        const result = this.create(matrix.cols, matrix.rows);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                result.data[j][i] = matrix.data[i][j];
            }
        }
        return result;
    }

    static softmax(matrix: Matrix): Matrix {
        const result = this.create(matrix.rows, matrix.cols);

        for (let i = 0; i < matrix.rows; i++) {
            // Find max for numerical stability
            let max = Math.max(...matrix.data[i]);
            let sum = 0;

            // Calculate exp and sum
            for (let j = 0; j < matrix.cols; j++) {
                result.data[i][j] = Math.exp(matrix.data[i][j] - max);
                sum += result.data[i][j];
            }

            // Normalize
            for (let j = 0; j < matrix.cols; j++) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }
}

// Layer Normalization
export class LayerNorm {
    gamma: Matrix;
    beta: Matrix;
    eps: number;

    constructor(features: number, eps = 1e-6) {
        this.gamma = MatrixOps.create(1, features, 1);
        this.beta = MatrixOps.create(1, features, 0);
        this.eps = eps;
    }

    forward(x: Matrix): Matrix {
        const result = MatrixOps.create(x.rows, x.cols);

        for (let i = 0; i < x.rows; i++) {
            // Calculate mean
            let mean = 0;
            for (let j = 0; j < x.cols; j++) {
                mean += x.data[i][j];
            }
            mean /= x.cols;

            // Calculate variance
            let variance = 0;
            for (let j = 0; j < x.cols; j++) {
                variance += Math.pow(x.data[i][j] - mean, 2);
            }
            variance /= x.cols;

            // Normalize
            const std = Math.sqrt(variance + this.eps);
            for (let j = 0; j < x.cols; j++) {
                const normalized = (x.data[i][j] - mean) / std;
                result.data[i][j] = this.gamma.data[0][j] * normalized + this.beta.data[0][j];
            }
        }

        return result;
    }
}

// Token Embeddings (simplified)
export class Embeddings {
    embeddingMatrix: Matrix;
    vocabSize: number;
    dModel: number;

    constructor(vocabSize: number, dModel: number) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.embeddingMatrix = MatrixOps.random(vocabSize, dModel, 0.1);
    }

    forward(tokenIds: number[]): Matrix {
        const seqLen = tokenIds.length;
        const embeddings = MatrixOps.create(seqLen, this.dModel);

        for (let i = 0; i < seqLen; i++) {
            const tokenId = tokenIds[i];
            for (let j = 0; j < this.dModel; j++) {
                embeddings.data[i][j] = this.embeddingMatrix.data[tokenId][j];
            }
        }

        return embeddings;
    }
}


// Multi-Head Attention
export class MultiHeadAttention {
    dModel: number;
    numHeads: number;
    dK: number;

    wQ: Matrix;
    wK: Matrix;
    wV: Matrix;
    wO: Matrix;

    constructor(dModel: number, numHeads: number) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = Math.floor(dModel / numHeads);

        // Initialize weight matrices
        this.wQ = MatrixOps.random(dModel, dModel);
        this.wK = MatrixOps.random(dModel, dModel);
        this.wV = MatrixOps.random(dModel, dModel);
        this.wO = MatrixOps.random(dModel, dModel);
    }

    forward(x: Matrix, mask?: Matrix): Matrix {
        const seqLen = x.rows;
        const batchSize = 1; // Simplified for this example

        // Linear projections
        const Q = MatrixOps.multiply(x, this.wQ);
        const K = MatrixOps.multiply(x, this.wK);
        const V = MatrixOps.multiply(x, this.wV);

        // Scaled dot-product attention (simplified single head)
        const scores = MatrixOps.multiply(Q, MatrixOps.transpose(K));

        // Scale by sqrt(d_k)
        const scale = 1.0 / Math.sqrt(this.dK);
        for (let i = 0; i < scores.rows; i++) {
            for (let j = 0; j < scores.cols; j++) {
                scores.data[i][j] *= scale;
            }
        }

        // Apply mask if provided (for causal attention)
        if (mask) {
            for (let i = 0; i < scores.rows; i++) {
                for (let j = 0; j < scores.cols; j++) {
                    if (mask.data[i][j] === 0) {
                        scores.data[i][j] = -Infinity;
                    }
                }
            }
        }

        // Apply softmax
        const attentionWeights = MatrixOps.softmax(scores);

        // Apply attention to values
        const output = MatrixOps.multiply(attentionWeights, V);

        // Final linear projection
        return MatrixOps.multiply(output, this.wO);
    }
}

// Feed Forward Network
export class FeedForward {
    w1: Matrix;
    b1: Matrix;
    w2: Matrix;
    b2: Matrix;

    constructor(dModel: number, dFf: number) {
        this.w1 = MatrixOps.random(dModel, dFf);
        this.b1 = MatrixOps.create(1, dFf, 0);
        this.w2 = MatrixOps.random(dFf, dModel);
        this.b2 = MatrixOps.create(1, dModel, 0);
    }

    forward(x: Matrix): Matrix {
        // First linear layer + activation
        let output = MatrixOps.multiply(x, this.w1);

        // Add bias (broadcast)
        for (let i = 0; i < output.rows; i++) {
            for (let j = 0; j < output.cols; j++) {
                output.data[i][j] += this.b1.data[0][j];
            }
        }

        // Apply GELU activation
        output = Activations.applyActivation(output, Activations.gelu);

        // Second linear layer
        output = MatrixOps.multiply(output, this.w2);

        // Add bias
        for (let i = 0; i < output.rows; i++) {
            for (let j = 0; j < output.cols; j++) {
                output.data[i][j] += this.b2.data[0][j];
            }
        }

        return output;
    }
}
