package nn

import (
	"fmt"
	"math/rand"
)

// Matrix represents a 2D matrix
type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{Rows: rows, Cols: cols, Data: data}
}

// RandomMatrix creates a matrix filled with random values
func RandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float64()*2 - 1 // Random values between -1 and 1
		}
	}
	return m
}

// Multiply performs matrix multiplication
func (m *Matrix) Multiply(other *Matrix) (*Matrix, error) {
	if m.Cols != other.Rows {
		return nil, fmt.Errorf("incompatible dimensions: (%d, %d) and (%d, %d)", m.Rows, m.Cols, other.Rows, other.Cols)
	}

	result := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result, nil
}

// Add performs element-wise addition
func (m *Matrix) Add(other *Matrix) (*Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, fmt.Errorf("incompatible dimensions")
	}

	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] + other.Data[i][j]
		}
	}
	return result, nil
}

// Scale multiplies all elements by a scalar
func (m *Matrix) Scale(scalar float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}
