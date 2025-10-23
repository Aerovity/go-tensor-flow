package nn

import (
	"fmt"
)

// Sequential model that stacks layers
type Sequential struct {
	Layers    []Layer
	Loss      Loss
	Optimizer Optimizer
}

// Optimizer interface for different optimization algorithms
type Optimizer interface {
	Update(paramName string, params, gradients *Matrix) *Matrix
}

// NewSequential creates a new sequential model
func NewSequential() *Sequential {
	return &Sequential{
		Layers: make([]Layer, 0),
	}
}

// Add adds a layer to the model
func (s *Sequential) Add(layer Layer) {
	s.Layers = append(s.Layers, layer)
}

// Compile sets the loss function and optimizer
func (s *Sequential) Compile(loss Loss, optimizer Optimizer) {
	s.Loss = loss
	s.Optimizer = optimizer
}

// Forward performs forward pass through all layers
func (s *Sequential) Forward(input *Matrix) (*Matrix, error) {
	output := input
	var err error

	for i, layer := range s.Layers {
		output, err = layer.Forward(output)
		if err != nil {
			return nil, fmt.Errorf("error in layer %d: %v", i, err)
		}
	}

	return output, nil
}

// Backward performs backward pass through all layers
func (s *Sequential) Backward(gradOutput *Matrix) error {
	grad := gradOutput

	// Backpropagate through layers in reverse order
	for i := len(s.Layers) - 1; i >= 0; i-- {
		var err error
		grad, err = s.Layers[i].Backward(grad)
		if err != nil {
			return fmt.Errorf("error in backward pass at layer %d: %v", i, err)
		}
	}

	return nil
}

// UpdateWeights updates all parameters using the optimizer
func (s *Sequential) UpdateWeights() {
	layerIdx := 0
	for _, layer := range s.Layers {
		params := layer.GetParams()
		grads := layer.GetGrads()
		paramNames := layer.GetParamNames()

		for i := range params {
			paramName := fmt.Sprintf("layer_%d_%s", layerIdx, paramNames[i])
			updated := s.Optimizer.Update(paramName, params[i], grads[i])

			// Update the parameter in place
			for r := 0; r < params[i].Rows; r++ {
				for c := 0; c < params[i].Cols; c++ {
					params[i].Data[r][c] = updated.Data[r][c]
				}
			}
		}
		layerIdx++
	}
}

// TrainOnBatch trains the model on a single batch
func (s *Sequential) TrainOnBatch(X, y *Matrix) (float64, error) {
	// Forward pass
	predictions, err := s.Forward(X)
	if err != nil {
		return 0, err
	}

	// Compute loss
	loss, err := s.Loss.Forward(predictions, y)
	if err != nil {
		return 0, err
	}

	// Backward pass through loss
	gradLoss, err := s.Loss.Backward(predictions, y)
	if err != nil {
		return 0, err
	}

	// Backward pass through layers
	err = s.Backward(gradLoss)
	if err != nil {
		return 0, err
	}

	// Update weights
	s.UpdateWeights()

	return loss, nil
}

// Fit trains the model for multiple epochs
func (s *Sequential) Fit(X, y *Matrix, epochs int, batchSize int, verbose bool) error {
	numSamples := X.Rows

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		numBatches := 0

		// Train on batches
		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples
			}

			// Create batch
			batchX := NewMatrix(end-i, X.Cols)
			batchY := NewMatrix(end-i, y.Cols)

			for j := i; j < end; j++ {
				for k := 0; k < X.Cols; k++ {
					batchX.Data[j-i][k] = X.Data[j][k]
				}
				for k := 0; k < y.Cols; k++ {
					batchY.Data[j-i][k] = y.Data[j][k]
				}
			}

			// Train on batch
			loss, err := s.TrainOnBatch(batchX, batchY)
			if err != nil {
				return err
			}

			totalLoss += loss
			numBatches++
		}

		avgLoss := totalLoss / float64(numBatches)

		if verbose {
			fmt.Printf("Epoch %d/%d - Loss: %.6f\n", epoch+1, epochs, avgLoss)
		}
	}

	return nil
}

// Predict makes predictions on input data
func (s *Sequential) Predict(X *Matrix) (*Matrix, error) {
	return s.Forward(X)
}

// Evaluate computes loss on test data
func (s *Sequential) Evaluate(X, y *Matrix) (float64, error) {
	predictions, err := s.Predict(X)
	if err != nil {
		return 0, err
	}

	loss, err := s.Loss.Forward(predictions, y)
	if err != nil {
		return 0, err
	}

	return loss, nil
}
