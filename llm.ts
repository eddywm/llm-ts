// LLM Transformer Model Implementation in TypeScript
// An experimental project on LLM transformer architecture.

import {Embeddings, FeedForward, LayerNorm, Matrix, MatrixOps, MultiHeadAttention } from "./matrix.ops";


// Transformer Block
class TransformerBlock {
    attention: MultiHeadAttention;
    feedForward: FeedForward;
    layerNorm1: LayerNorm;
    layerNorm2: LayerNorm;

    constructor(dModel: number, numHeads: number, dFf: number) {
        this.attention = new MultiHeadAttention(dModel, numHeads);
        this.feedForward = new FeedForward(dModel, dFf);
        this.layerNorm1 = new LayerNorm(dModel);
        this.layerNorm2 = new LayerNorm(dModel);
    }

    forward(x: Matrix, mask?: Matrix): Matrix {
        // Self-attention with residual connection and layer norm
        const attnOutput = this.attention.forward(x, mask);
        const x1 = MatrixOps.add(x, attnOutput);
        const normalized1 = this.layerNorm1.forward(x1);

        // Feed forward with residual connection and layer norm
        const ffOutput = this.feedForward.forward(normalized1);
        const x2 = MatrixOps.add(normalized1, ffOutput);
        return this.layerNorm2.forward(x2);
    }
}

// Positional Encoding
class PositionalEncoding {
    static generate(seqLen: number, dModel: number): Matrix {
        const pe = MatrixOps.create(seqLen, dModel);

        for (let pos = 0; pos < seqLen; pos++) {
            for (let i = 0; i < dModel; i += 2) {
                const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
                pe.data[pos][i] = Math.sin(angle);
                if (i + 1 < dModel) {
                    pe.data[pos][i + 1] = Math.cos(angle);
                }
            }
        }

        return pe;
    }
}



// Simple Tokenizer (character-level for demo)
class SimpleTokenizer {
    charToId: Map<string, number> = new Map();
    idToChar: Map<number, string> = new Map();
    vocabSize: number = 0;

    constructor(text: string) {
        const uniqueChars = Array.from(new Set(text));
        uniqueChars.forEach((char, idx) => {
            this.charToId.set(char, idx);
            this.idToChar.set(idx, char);
        });
        this.vocabSize = uniqueChars.length;
    }

    encode(text: string): number[] {
        return text.split('').map(char => this.charToId.get(char) || 0);
    }

    decode(tokens: number[]): string {
        return tokens.map(token => this.idToChar.get(token) || '').join('');
    }
}

// Main Transformer Model
class TransformerLLM {
    embeddings: Embeddings;
    positionalEncoding: Matrix;
    blocks: TransformerBlock[];
    layerNorm: LayerNorm;
    outputProjection: Matrix;
    tokenizer: SimpleTokenizer;

    dModel: number;
    numBlocks: number;
    numHeads: number;
    maxSeqLen: number;

    constructor(
        vocabSize: number,
        dModel: number = 128,
        numBlocks: number = 4,
        numHeads: number = 8,
        dFf: number = 512,
        maxSeqLen: number = 256
    ) {
        this.dModel = dModel;
        this.numBlocks = numBlocks;
        this.numHeads = numHeads;
        this.maxSeqLen = maxSeqLen;

        // Initialize components
        this.embeddings = new Embeddings(vocabSize, dModel);
        this.positionalEncoding = PositionalEncoding.generate(maxSeqLen, dModel);

        this.blocks = [];
        for (let i = 0; i < numBlocks; i++) {
            this.blocks.push(new TransformerBlock(dModel, numHeads, dFf));
        }

        this.layerNorm = new LayerNorm(dModel);
        this.outputProjection = MatrixOps.random(dModel, vocabSize);
    }

    createCausalMask(seqLen: number): Matrix {
        const mask = MatrixOps.create(seqLen, seqLen);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j <= i; j++) {
                mask.data[i][j] = 1;
            }
        }
        return mask;
    }

    forward(tokenIds: number[]): Matrix {
        const seqLen = tokenIds.length;

        // Token embeddings
        let x = this.embeddings.forward(tokenIds);

        // Add positional encoding
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < this.dModel; j++) {
                x.data[i][j] += this.positionalEncoding.data[i][j];
            }
        }

        // Create causal mask
        const mask = this.createCausalMask(seqLen);

        // Pass through transformer blocks
        for (const block of this.blocks) {
            x = block.forward(x, mask);
        }

        // Final layer normalization
        x = this.layerNorm.forward(x);

        // Output projection to vocabulary
        const logits = MatrixOps.multiply(x, this.outputProjection);

        return logits;
    }

    predict(text: string, maxNewTokens: number = 10): string {
        if (!this.tokenizer) {
            throw new Error('Tokenizer not set. Please set a tokenizer first.');
        }

        let tokens = this.tokenizer.encode(text);

        for (let i = 0; i < maxNewTokens; i++) {
            // Get logits for current sequence
            const logits = this.forward(tokens);

            // Get logits for the last position (next token prediction)
            const lastLogits = logits.data[logits.rows - 1];

            // Simple greedy sampling (choose argmax)
            let maxIdx = 0;
            let maxVal = lastLogits[0];
            for (let j = 1; j < lastLogits.length; j++) {
                if (lastLogits[j] > maxVal) {
                    maxVal = lastLogits[j];
                    maxIdx = j;
                }
            }

            // Add predicted token
            tokens.push(maxIdx);

            // Stop if we hit max sequence length
            if (tokens.length >= this.maxSeqLen) {
                break;
            }
        }

        return this.tokenizer.decode(tokens);
    }

    setTokenizer(tokenizer: SimpleTokenizer): void {
        this.tokenizer = tokenizer;
    }
}

// ============================================================================
// USAGE EXAMPLES AND DEMONSTRATIONS
// ============================================================================

console.log("🚀 LLM Transformer Model Demo\n");

// Example 1: Create a simple model and demonstrate components
console.log("📚 Example 1: Basic Model Creation and Component Demonstration");
console.log("=" .repeat(70));

const sampleText = "hello world this is a simple example of text generation using transformers";
const tokenizer = new SimpleTokenizer(sampleText);

console.log(`Vocabulary size: ${tokenizer.vocabSize}`);
console.log(`Sample encoding: "${sampleText.slice(0, 10)}" -> [${tokenizer.encode(sampleText.slice(0, 10)).join(', ')}]`);

// Create the model
const model = new TransformerLLM(
    tokenizer.vocabSize,  // vocab size
    64,                   // d_model (smaller for demo)
    2,                    // num blocks
    4,                    // num heads
    256,                  // d_ff
    128                   // max sequence length
);

model.setTokenizer(tokenizer);

console.log(`\nModel created with:`);
console.log(`- Embedding dimension: ${model.dModel}`);
console.log(`- Number of blocks: ${model.numBlocks}`);
console.log(`- Number of attention heads: ${model.numHeads}\n`);

// Example 2: Forward pass demonstration
console.log("🔍 Example 2: Forward Pass Demonstration");
console.log("=" .repeat(70));

const inputText = "hello";
const inputTokens = tokenizer.encode(inputText);
console.log(`Input: "${inputText}"`);
console.log(`Tokens: [${inputTokens.join(', ')}]`);

const logits = model.forward(inputTokens);
console.log(`Output logits shape: ${logits.rows} x ${logits.cols}`);
console.log(`Sample logits for first position: [${logits.data[0].slice(0, 5).map(x => x.toFixed(3)).join(', ')}...]`);

// Example 3: Text generation
console.log("\n🎯 Example 3: Text Generation");
console.log("=" .repeat(70));

const promptText = "hello";
console.log(`Prompt: "${promptText}"`);

try {
    const generated = model.predict(promptText, 5);
    console.log(`Generated: "${generated}"`);
    console.log(`New tokens added: "${generated.slice(promptText.length)}"`);
} catch (error) {
    console.log(`Generation result: ${error.message}`);
}

// Example 4: Component-by-component breakdown
console.log("\n🔧 Example 4: Component Breakdown");
console.log("=" .repeat(70));

// Demonstrate embeddings
const testTokens = [0, 1, 2];
const embeddings = new Embeddings(tokenizer.vocabSize, 32);
const embedded = embeddings.forward(testTokens);
console.log(`Token embeddings shape: ${embedded.rows} x ${embedded.cols}`);

// Demonstrate positional encoding
const posEncoding = PositionalEncoding.generate(5, 32);
console.log(`Positional encoding shape: ${posEncoding.rows} x ${posEncoding.cols}`);
console.log(`Sample positional encoding: [${posEncoding.data[0].slice(0, 4).map(x => x.toFixed(3)).join(', ')}...]`);

// Demonstrate attention
const attention = new MultiHeadAttention(32, 4);
const testMatrix = MatrixOps.random(3, 32, 0.1);
const attentionOutput = attention.forward(testMatrix);
console.log(`Attention output shape: ${attentionOutput.rows} x ${attentionOutput.cols}`);

// Example 5: Matrix operations demonstration
console.log("\n🧮 Example 5: Matrix Operations");
console.log("=" .repeat(70));

const matA = MatrixOps.create(2, 3);
matA.data = [[1, 2, 3], [4, 5, 6]];

const matB = MatrixOps.create(3, 2);
matB.data = [[1, 2], [3, 4], [5, 6]];

const product = MatrixOps.multiply(matA, matB);
console.log("Matrix A:", matA.data);
console.log("Matrix B:", matB.data);
console.log("A × B =", product.data);

// Demonstrate softmax
const testScores = MatrixOps.create(1, 4);
testScores.data = [[1, 2, 3, 4]];
const softmaxOutput = MatrixOps.softmax(testScores);
console.log(`Softmax input: [${testScores.data[0].join(', ')}]`);
console.log(`Softmax output: [${softmaxOutput.data[0].map(x => x.toFixed(3)).join(', ')}]`);
console.log(`Sum: ${softmaxOutput.data[0].reduce((a, b) => a + b, 0).toFixed(3)}`);

// Example 6: Training simulation (simplified)
console.log("\n🏋️ Example 6: Model Analysis");
console.log("=" .repeat(70));

console.log("Model parameter count estimate:");
let totalParams = 0;

// Embeddings
const embeddingParams = tokenizer.vocabSize * model.dModel;
totalParams += embeddingParams;
console.log(`- Embeddings: ${embeddingParams.toLocaleString()}`);

// Each transformer block
const attentionParams = 4 * model.dModel * model.dModel; // Q, K, V, O projections
const feedForwardParams = model.dModel * 256 + 256 * model.dModel; // FFN layers
const blockParams = attentionParams + feedForwardParams + 2 * model.dModel; // + layer norms
const allBlocksParams = blockParams * model.numBlocks;
totalParams += allBlocksParams;
console.log(`- Transformer blocks (${model.numBlocks}): ${allBlocksParams.toLocaleString()}`);

// Output projection
const outputParams = model.dModel * tokenizer.vocabSize;
totalParams += outputParams;
console.log(`- Output projection: ${outputParams.toLocaleString()}`);

console.log(`- Total parameters: ~${totalParams.toLocaleString()}`);

console.log("\n✅ Demo completed! The transformer model includes:");
console.log("- Token embeddings and positional encoding");
console.log("- Multi-head self-attention mechanism");
console.log("- Feed-forward networks with GELU activation");
console.log("- Layer normalization and residual connections");
console.log("- Causal masking for autoregressive generation");
console.log("- Matrix operations optimized for transformer architecture");

console.log("\n💡 Note: This is an experimental implementation!");
console.log("Production models use optimizations like:");
console.log("- GPU acceleration");
console.log("- More sophisticated tokenization (BPE/SentencePiece)");
console.log("- Advanced training techniques (gradient clipping, learning rate scheduling)");
console.log("- Larger model dimensions and more sophisticated architectures");