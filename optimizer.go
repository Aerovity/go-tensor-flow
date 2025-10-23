package nn

import "math"

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64 // Exponential decay rate for first moment estimates
	Beta2        float64 // Exponential decay rate for second moment estimates
	Epsilon      float64 // Small constant for numerical stability
	T            int     // Time step

	// First moment vector (mean of gradients)
	M map[string]*Matrix

	// Second moment vector (uncentered variance of gradients)
	V map[string]*Matrix
}

// NewAdamOptimizer creates a new Adam optimizer with default parameters
func NewAdamOptimizer(learningRate float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		T:            0,
		M:            make(map[string]*Matrix),
		V:            make(map[string]*Matrix),
	}
}

// Update updates parameters using Adam algorithm
func (adam *AdamOptimizer) Update(paramName string, params, gradients *Matrix) *Matrix {
	adam.T++

	// Initialize moment vectors if not exists
	if adam.M[paramName] == nil {
		adam.M[paramName] = NewMatrix(params.Rows, params.Cols)
		adam.V[paramName] = NewMatrix(params.Rows, params.Cols)
	}

	m := adam.M[paramName]
	v := adam.V[paramName]

	// Update biased first moment estimate
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			m.Data[i][j] = adam.Beta1*m.Data[i][j] + (1-adam.Beta1)*gradients.Data[i][j]
		}
	}

	// Update biased second raw moment estimate
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			v.Data[i][j] = adam.Beta2*v.Data[i][j] + (1-adam.Beta2)*gradients.Data[i][j]*gradients.Data[i][j]
		}
	}

	// Compute bias-corrected first moment estimate
	mHat := NewMatrix(params.Rows, params.Cols)
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			mHat.Data[i][j] = m.Data[i][j] / (1 - math.Pow(adam.Beta1, float64(adam.T)))
		}
	}

	// Compute bias-corrected second raw moment estimate
	vHat := NewMatrix(params.Rows, params.Cols)
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			vHat.Data[i][j] = v.Data[i][j] / (1 - math.Pow(adam.Beta2, float64(adam.T)))
		}
	}

	// Update parameters
	updated := NewMatrix(params.Rows, params.Cols)
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			updated.Data[i][j] = params.Data[i][j] - adam.LearningRate*mHat.Data[i][j]/(math.Sqrt(vHat.Data[i][j])+adam.Epsilon)
		}
	}

	return updated
}

// SGD implements simple stochastic gradient descent
type SGD struct {
	LearningRate float64
	Momentum     float64
	Velocity     map[string]*Matrix
}

// NewSGD creates a new SGD optimizer
func NewSGD(learningRate, momentum float64) *SGD {
	return &SGD{
		LearningRate: learningRate,
		Momentum:     momentum,
		Velocity:     make(map[string]*Matrix),
	}
}

// Update updates parameters using SGD with momentum
func (sgd *SGD) Update(paramName string, params, gradients *Matrix) *Matrix {
	// Initialize velocity if not exists
	if sgd.Velocity[paramName] == nil {
		sgd.Velocity[paramName] = NewMatrix(params.Rows, params.Cols)
	}

	velocity := sgd.Velocity[paramName]

	// Update velocity and parameters
	updated := NewMatrix(params.Rows, params.Cols)
	for i := 0; i < params.Rows; i++ {
		for j := 0; j < params.Cols; j++ {
			velocity.Data[i][j] = sgd.Momentum*velocity.Data[i][j] - sgd.LearningRate*gradients.Data[i][j]
			updated.Data[i][j] = params.Data[i][j] + velocity.Data[i][j]
		}
	}

	return updated
}
