package main

import (
	"fmt"
	"math/rand"
	"time"

	nn "github.com/Aerovity/go-tensor-flow"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Example 1: Simple Binary Classification
	fmt.Println("=== Binary Classification Example ===")
	binaryClassificationExample()

	fmt.Println("\n=== Multi-class Classification Example ===")
	multiClassExample()

	fmt.Println("\n=== Regression Example ===")
	regressionExample()
}

func binaryClassificationExample() {
	// Create simple dataset: XOR problem
	// Input: 2 features, Output: 1 (binary)
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
	model.Add(nn.NewDense(2, 8))   // Input layer: 2 -> 8
	model.Add(nn.NewReLULayer())    // Activation
	model.Add(nn.NewDense(8, 4))    // Hidden layer: 8 -> 4
	model.Add(nn.NewReLULayer())    // Activation
	model.Add(nn.NewDense(4, 1))    // Output layer: 4 -> 1

	// Compile model
	model.Compile(
		nn.NewBinaryCrossEntropy(),
		nn.NewAdamOptimizer(0.01),
	)

	// Train model
	fmt.Println("Training...")
	err := model.Fit(X, y, 1000, 4, true)
	if err != nil {
		fmt.Printf("Error during training: %v\n", err)
		return
	}

	// Make predictions
	fmt.Println("\nPredictions:")
	predictions, _ := model.Predict(X)
	for i := 0; i < 4; i++ {
		fmt.Printf("Input: [%.0f, %.0f] -> Predicted: %.4f, Actual: %.0f\n",
			X.Data[i][0], X.Data[i][1], predictions.Data[i][0], y.Data[i][0])
	}
}

func multiClassExample() {
	// Create a simple 3-class classification problem
	// Each class has 10 samples
	numClasses := 3
	samplesPerClass := 10
	totalSamples := numClasses * samplesPerClass

	X := nn.NewMatrix(totalSamples, 2)
	y := nn.NewMatrix(totalSamples, numClasses)

	// Generate synthetic data
	idx := 0
	for class := 0; class < numClasses; class++ {
		for i := 0; i < samplesPerClass; i++ {
			// Create cluster for each class
			X.Data[idx][0] = float64(class) + rand.Float64()*0.5
			X.Data[idx][1] = float64(class) + rand.Float64()*0.5

			// One-hot encode labels
			y.Data[idx][class] = 1.0

			idx++
		}
	}

	// Build model
	model := nn.NewSequential()
	model.Add(nn.NewDense(2, 16))
	model.Add(nn.NewReLULayer())
	model.Add(nn.NewDense(16, 8))
	model.Add(nn.NewReLULayer())
	model.Add(nn.NewDense(8, numClasses))
	model.Add(nn.NewSoftmaxLayer())

	// Compile
	model.Compile(
		nn.NewCategoricalCrossEntropy(),
		nn.NewAdamOptimizer(0.01),
	)

	// Train
	fmt.Println("Training multi-class model...")
	err := model.Fit(X, y, 500, 10, true)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Evaluate
	loss, _ := model.Evaluate(X, y)
	fmt.Printf("\nFinal Loss: %.6f\n", loss)
}

func regressionExample() {
	// Simple linear regression: y = 2x + 1 + noise
	numSamples := 100
	X := nn.NewMatrix(numSamples, 1)
	y := nn.NewMatrix(numSamples, 1)

	for i := 0; i < numSamples; i++ {
		x := rand.Float64() * 10
		X.Data[i][0] = x
		y.Data[i][0] = 2*x + 1 + rand.NormFloat64()*0.5
	}

	// Build model
	model := nn.NewSequential()
	model.Add(nn.NewDense(1, 8))
	model.Add(nn.NewReLULayer())
	model.Add(nn.NewDense(8, 1))

	// Compile with MSE loss
	model.Compile(
		nn.NewMSE(),
		nn.NewAdamOptimizer(0.01),
	)

	// Train
	fmt.Println("Training regression model...")
	err := model.Fit(X, y, 200, 20, true)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Test predictions
	testX := nn.NewMatrix(5, 1)
	testX.Data[0][0] = 1.0
	testX.Data[1][0] = 2.0
	testX.Data[2][0] = 3.0
	testX.Data[3][0] = 4.0
	testX.Data[4][0] = 5.0

	predictions, _ := model.Predict(testX)
	fmt.Println("\nRegression predictions (y ≈ 2x + 1):")
	for i := 0; i < 5; i++ {
		expected := 2*testX.Data[i][0] + 1
		fmt.Printf("x=%.0f -> predicted: %.2f, expected: %.2f\n",
			testX.Data[i][0], predictions.Data[i][0], expected)
	}
}
