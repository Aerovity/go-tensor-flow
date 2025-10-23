package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// Layer interface for neural network layers
type Layer interface {
	Forward(input *Matrix) (*Matrix, error)
	Backward(gradOutput *Matrix) (*Matrix, error)
	GetParams() []*Matrix
	GetGrads() []*Matrix
	GetParamNames() []string
}

// Dense (Fully Connected) Layer
type Dense struct {
	InputSize  int
	OutputSize int
	Weights    *Matrix // Shape: (InputSize, OutputSize)
	Bias       *Matrix // Shape: (1, OutputSize)

	// Cache for backward pass
	lastInput   *Matrix
	weightsGrad *Matrix
	biasGrad    *Matrix
}

// NewDense creates a new dense layer with He initialization
func NewDense(inputSize, outputSize int) *Dense {
	// He initialization: scale by sqrt(2/inputSize)
	weights := NewMatrix(inputSize, outputSize)
	scale := math.Sqrt(2.0 / float64(inputSize))
	for i := 0; i < inputSize; i++ {
		for j := 0; j < outputSize; j++ {
			weights.Data[i][j] = rand.NormFloat64() * scale
		}
	}

	bias := NewMatrix(1, outputSize)

	return &Dense{
		InputSize:   inputSize,
		OutputSize:  outputSize,
		Weights:     weights,
		Bias:        bias,
		weightsGrad: NewMatrix(inputSize, outputSize),
		biasGrad:    NewMatrix(1, outputSize),
	}
}

// Forward performs forward pass: output = input @ weights + bias
func (d *Dense) Forward(input *Matrix) (*Matrix, error) {
	if input.Cols != d.InputSize {
		return nil, fmt.Errorf("input size mismatch: got %d, expected %d", input.Cols, d.InputSize)
	}

	d.lastInput = input

	// Matrix multiplication
	output, err := input.Multiply(d.Weights)
	if err != nil {
		return nil, err
	}

	// Add bias to each row
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			output.Data[i][j] += d.Bias.Data[0][j]
		}
	}

	return output, nil
}

// Backward computes gradients for backpropagation
func (d *Dense) Backward(gradOutput *Matrix) (*Matrix, error) {
	if gradOutput.Cols != d.OutputSize {
		return nil, fmt.Errorf("gradient size mismatch")
	}

	batchSize := float64(gradOutput.Rows)

	// Compute weight gradient: dL/dW = input^T @ gradOutput
	// weightsGrad shape: (InputSize, OutputSize)
	for i := 0; i < d.InputSize; i++ {
		for j := 0; j < d.OutputSize; j++ {
			sum := 0.0
			for b := 0; b < d.lastInput.Rows; b++ {
				sum += d.lastInput.Data[b][i] * gradOutput.Data[b][j]
			}
			d.weightsGrad.Data[i][j] = sum / batchSize
		}
	}

	// Compute bias gradient: sum over batch dimension
	for j := 0; j < d.OutputSize; j++ {
		sum := 0.0
		for i := 0; i < gradOutput.Rows; i++ {
			sum += gradOutput.Data[i][j]
		}
		d.biasGrad.Data[0][j] = sum / batchSize
	}

	// Compute input gradient: gradOutput @ weights^T
	gradInput := NewMatrix(gradOutput.Rows, d.InputSize)
	for i := 0; i < gradOutput.Rows; i++ {
		for j := 0; j < d.InputSize; j++ {
			sum := 0.0
			for k := 0; k < d.OutputSize; k++ {
				sum += gradOutput.Data[i][k] * d.Weights.Data[j][k]
			}
			gradInput.Data[i][j] = sum
		}
	}

	return gradInput, nil
}

// GetParams returns the parameters of the layer
func (d *Dense) GetParams() []*Matrix {
	return []*Matrix{d.Weights, d.Bias}
}

// GetGrads returns the gradients of the parameters
func (d *Dense) GetGrads() []*Matrix {
	return []*Matrix{d.weightsGrad, d.biasGrad}
}

// GetParamNames returns names for the parameters
func (d *Dense) GetParamNames() []string {
	return []string{"weights", "bias"}
}

// ReLULayer activation layer
type ReLULayer struct {
	lastInput *Matrix
}

// NewReLULayer creates a new ReLU activation layer
func NewReLULayer() *ReLULayer {
	return &ReLULayer{}
}

// Forward applies ReLU activation
func (r *ReLULayer) Forward(input *Matrix) (*Matrix, error) {
	r.lastInput = input
	return ReLUMatrix(input), nil
}

// Backward computes gradient for ReLU
func (r *ReLULayer) Backward(gradOutput *Matrix) (*Matrix, error) {
	gradInput := NewMatrix(gradOutput.Rows, gradOutput.Cols)
	for i := 0; i < gradOutput.Rows; i++ {
		for j := 0; j < gradOutput.Cols; j++ {
			if r.lastInput.Data[i][j] > 0 {
				gradInput.Data[i][j] = gradOutput.Data[i][j]
			}
		}
	}
	return gradInput, nil
}

// GetParams returns empty slice (no learnable parameters)
func (r *ReLULayer) GetParams() []*Matrix {
	return []*Matrix{}
}

// GetGrads returns empty slice
func (r *ReLULayer) GetGrads() []*Matrix {
	return []*Matrix{}
}

// GetParamNames returns empty slice
func (r *ReLULayer) GetParamNames() []string {
	return []string{}
}

// SoftmaxLayer activation layer
type SoftmaxLayer struct {
	lastOutput *Matrix
}

// NewSoftmaxLayer creates a new softmax layer
func NewSoftmaxLayer() *SoftmaxLayer {
	return &SoftmaxLayer{}
}

// Forward applies softmax activation
func (s *SoftmaxLayer) Forward(input *Matrix) (*Matrix, error) {
	s.lastOutput = SoftmaxMatrix(input)
	return s.lastOutput, nil
}

// Backward computes gradient for softmax (simplified for use with cross-entropy)
func (s *SoftmaxLayer) Backward(gradOutput *Matrix) (*Matrix, error) {
	// When used with cross-entropy, the gradient is simply passed through
	// The actual softmax gradient is handled in the loss function
	return gradOutput, nil
}

// GetParams returns empty slice
func (s *SoftmaxLayer) GetParams() []*Matrix {
	return []*Matrix{}
}

// GetGrads returns empty slice
func (s *SoftmaxLayer) GetGrads() []*Matrix {
	return []*Matrix{}
}

// GetParamNames returns empty slice
func (s *SoftmaxLayer) GetParamNames() []string {
	return []string{}
}
