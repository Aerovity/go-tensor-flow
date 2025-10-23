package nn

import (
	"fmt"
	"math"
)

// Loss interface for different loss functions
type Loss interface {
	Forward(predictions, targets *Matrix) (float64, error)
	Backward(predictions, targets *Matrix) (*Matrix, error)
}

// BinaryCrossEntropy loss for binary classification
type BinaryCrossEntropy struct {
	Epsilon float64 // Small value to avoid log(0)
}

// NewBinaryCrossEntropy creates a new binary cross-entropy loss
func NewBinaryCrossEntropy() *BinaryCrossEntropy {
	return &BinaryCrossEntropy{Epsilon: 1e-7}
}

// Forward computes the binary cross-entropy loss
// L = -1/N * Σ(y*log(p) + (1-y)*log(1-p))
func (bce *BinaryCrossEntropy) Forward(predictions, targets *Matrix) (float64, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return 0, fmt.Errorf("shape mismatch: predictions %dx%d, targets %dx%d",
			predictions.Rows, predictions.Cols, targets.Rows, targets.Cols)
	}

	totalLoss := 0.0
	n := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			pred := math.Max(bce.Epsilon, math.Min(1-bce.Epsilon, predictions.Data[i][j]))
			target := targets.Data[i][j]
			totalLoss += -(target*math.Log(pred) + (1-target)*math.Log(1-pred))
		}
	}

	return totalLoss / n, nil
}

// Backward computes the gradient of binary cross-entropy
// dL/dp = -1/N * (y/p - (1-y)/(1-p))
func (bce *BinaryCrossEntropy) Backward(predictions, targets *Matrix) (*Matrix, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return nil, fmt.Errorf("shape mismatch")
	}

	gradient := NewMatrix(predictions.Rows, predictions.Cols)
	n := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			pred := math.Max(bce.Epsilon, math.Min(1-bce.Epsilon, predictions.Data[i][j]))
			target := targets.Data[i][j]
			gradient.Data[i][j] = -(target/pred - (1-target)/(1-pred)) / n
		}
	}

	return gradient, nil
}

// CategoricalCrossEntropy loss for multi-class classification
type CategoricalCrossEntropy struct {
	Epsilon float64
}

// NewCategoricalCrossEntropy creates a new categorical cross-entropy loss
func NewCategoricalCrossEntropy() *CategoricalCrossEntropy {
	return &CategoricalCrossEntropy{Epsilon: 1e-7}
}

// Forward computes the categorical cross-entropy loss
// L = -1/N * Σ(y*log(p))
func (cce *CategoricalCrossEntropy) Forward(predictions, targets *Matrix) (float64, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return 0, fmt.Errorf("shape mismatch")
	}

	totalLoss := 0.0
	n := float64(predictions.Rows)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			pred := math.Max(cce.Epsilon, predictions.Data[i][j])
			totalLoss += -targets.Data[i][j] * math.Log(pred)
		}
	}

	return totalLoss / n, nil
}

// Backward computes the gradient of categorical cross-entropy
// dL/dp = -y/p
func (cce *CategoricalCrossEntropy) Backward(predictions, targets *Matrix) (*Matrix, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return nil, fmt.Errorf("shape mismatch")
	}

	gradient := NewMatrix(predictions.Rows, predictions.Cols)
	n := float64(predictions.Rows)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			pred := math.Max(cce.Epsilon, predictions.Data[i][j])
			gradient.Data[i][j] = -targets.Data[i][j] / pred / n
		}
	}

	return gradient, nil
}

// MSE (Mean Squared Error) loss for regression
type MSE struct{}

// NewMSE creates a new MSE loss
func NewMSE() *MSE {
	return &MSE{}
}

// Forward computes mean squared error
// L = 1/(2N) * Σ(y - p)²
func (mse *MSE) Forward(predictions, targets *Matrix) (float64, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return 0, fmt.Errorf("shape mismatch")
	}

	totalLoss := 0.0
	n := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			diff := targets.Data[i][j] - predictions.Data[i][j]
			totalLoss += diff * diff
		}
	}

	return totalLoss / (2 * n), nil
}

// Backward computes the gradient of MSE
// dL/dp = -(y - p) / N
func (mse *MSE) Backward(predictions, targets *Matrix) (*Matrix, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return nil, fmt.Errorf("shape mismatch")
	}

	gradient := NewMatrix(predictions.Rows, predictions.Cols)
	n := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			gradient.Data[i][j] = -(targets.Data[i][j] - predictions.Data[i][j]) / n
		}
	}

	return gradient, nil
}
