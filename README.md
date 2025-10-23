# Go TensorFlow - Simple Neural Network Library

A lightweight neural network library written in Go for educational purposes and simple ML projects.

## Features

- ✅ **Dense (Fully Connected) Layers** with He initialization
- ✅ **Activation Functions**: ReLU, Softmax
- ✅ **Loss Functions**: Binary Cross-Entropy, Categorical Cross-Entropy, MSE
- ✅ **Optimizers**: Adam (with bias correction), SGD with momentum
- ✅ **CNN Support**: Convolutional layers and Max Pooling
- ✅ **Sequential Model API**: Easy layer stacking and training

## Installation

```bash
go get github.com/Aerovity/go-tensor-flow
```

## Quick Start

### Binary Classification Example

```go
package main

import (
    nn "github.com/Aerovity/go-tensor-flow"
)

func main() {
    // Create dataset
    X := nn.NewMatrix(4, 2)
    X.Data[0] = []float64{0, 0}
    X.Data[1] = []float64{0, 1}
    X.Data[2] = []float64{1, 0}
    X.Data[3] = []float64{1, 1}

    y := nn.NewMatrix(4, 1)
    y.Data[0] = []float64{0}
    y.Data[1] = []float64{1}
    y.Data[2] = []float64{1}
    y.Data[3] = []float64{0}

    // Build model
    model := nn.NewSequential()
    model.Add(nn.NewDense(2, 8))
    model.Add(nn.NewReLULayer())
    model.Add(nn.NewDense(8, 1))

    // Compile with loss and optimizer
    model.Compile(
        nn.NewBinaryCrossEntropy(),
        nn.NewAdamOptimizer(0.01),
    )

    // Train
    model.Fit(X, y, 1000, 4, true)

    // Predict
    predictions, _ := model.Predict(X)
}
```

### Multi-class Classification

```go
model := nn.NewSequential()
model.Add(nn.NewDense(inputSize, 64))
model.Add(nn.NewReLULayer())
model.Add(nn.NewDense(64, numClasses))
model.Add(nn.NewSoftmaxLayer())

model.Compile(
    nn.NewCategoricalCrossEntropy(),
    nn.NewAdamOptimizer(0.001),
)

model.Fit(X, y, epochs, batchSize, verbose)
```

### Regression

```go
model := nn.NewSequential()
model.Add(nn.NewDense(inputSize, 32))
model.Add(nn.NewReLULayer())
model.Add(nn.NewDense(32, 1))

model.Compile(
    nn.NewMSE(),
    nn.NewSGD(0.01, 0.9), // SGD with momentum
)

model.Fit(X, y, epochs, batchSize, verbose)
```

## API Reference

### Creating Models

```go
model := nn.NewSequential()
```

### Adding Layers

```go
model.Add(nn.NewDense(inputSize, outputSize))  // Fully connected layer
model.Add(nn.NewReLULayer())                    // ReLU activation
model.Add(nn.NewSoftmaxLayer())                 // Softmax activation
```

### Loss Functions

```go
nn.NewBinaryCrossEntropy()      // For binary classification
nn.NewCategoricalCrossEntropy()  // For multi-class classification
nn.NewMSE()                      // For regression
```

### Optimizers

```go
nn.NewAdamOptimizer(learningRate)           // Adam optimizer
nn.NewSGD(learningRate, momentum)           // SGD with momentum
```

### Training

```go
model.Compile(loss, optimizer)
model.Fit(X, y, epochs, batchSize, verbose)
```

### Prediction & Evaluation

```go
predictions, err := model.Predict(X)
loss, err := model.Evaluate(X, y)
```

## CNN Example

```go
// Coming soon: Full CNN integration with Sequential API
convLayer := nn.NewConvLayer(numFilters, inChannels, filterSize, stride, padding)
output, _ := convLayer.Forward(input3D)

poolLayer := nn.NewMaxPool2D(poolSize, stride)
pooled := poolLayer.Forward(output)
```

## Running Examples

```bash
cd example
go run main.go
```

## License

MIT License
