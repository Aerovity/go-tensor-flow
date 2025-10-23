package nn

import (
	"math"
)

// Softmax applies the softmax activation function
// Converts a vector of values into a probability distribution
func Softmax(input []float64) []float64 {
	// Find max for numerical stability
	max := input[0]
	for _, v := range input {
		if v > max {
			max = v
		}
	}

	// Compute exp(x - max) and sum
	expValues := make([]float64, len(input))
	sum := 0.0
	for i, v := range input {
		expValues[i] = math.Exp(v - max)
		sum += expValues[i]
	}

	// Normalize
	output := make([]float64, len(input))
	for i := range expValues {
		output[i] = expValues[i] / sum
	}

	return output
}

// SoftmaxMatrix applies softmax to each row of a matrix
func SoftmaxMatrix(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		result.Data[i] = Softmax(m.Data[i])
	}
	return result
}

// ReLU applies the ReLU activation function
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUMatrix applies ReLU to all elements
func ReLUMatrix(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = ReLU(m.Data[i][j])
		}
	}
	return result
}
