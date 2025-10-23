package nn

import (
	"fmt"
	"math/rand"
)

// Tensor3D represents a 3D tensor (channels, height, width)
type Tensor3D struct {
	Channels int
	Height   int
	Width    int
	Data     [][][]float64
}

// NewTensor3D creates a new 3D tensor
func NewTensor3D(channels, height, width int) *Tensor3D {
	data := make([][][]float64, channels)
	for c := range data {
		data[c] = make([][]float64, height)
		for h := range data[c] {
			data[c][h] = make([]float64, width)
		}
	}
	return &Tensor3D{Channels: channels, Height: height, Width: width, Data: data}
}

// ConvLayer represents a convolutional layer
type ConvLayer struct {
	NumFilters  int
	FilterSize  int
	Stride      int
	Padding     int
	InChannels  int
	Filters     [][][][]float64 // [numFilters][inChannels][filterSize][filterSize]
	Bias        []float64
}

// NewConvLayer creates a new convolutional layer
func NewConvLayer(numFilters, inChannels, filterSize, stride, padding int) *ConvLayer {
	// Initialize filters with small random values
	filters := make([][][][]float64, numFilters)
	for f := 0; f < numFilters; f++ {
		filters[f] = make([][][]float64, inChannels)
		for c := 0; c < inChannels; c++ {
			filters[f][c] = make([][]float64, filterSize)
			for i := 0; i < filterSize; i++ {
				filters[f][c][i] = make([]float64, filterSize)
				for j := 0; j < filterSize; j++ {
					filters[f][c][i][j] = (rand.Float64()*2 - 1) * 0.1
				}
			}
		}
	}

	bias := make([]float64, numFilters)

	return &ConvLayer{
		NumFilters:  numFilters,
		FilterSize:  filterSize,
		Stride:      stride,
		Padding:     padding,
		InChannels:  inChannels,
		Filters:     filters,
		Bias:        bias,
	}
}

// Forward performs the forward pass of convolution
func (conv *ConvLayer) Forward(input *Tensor3D) (*Tensor3D, error) {
	if input.Channels != conv.InChannels {
		return nil, fmt.Errorf("input channels mismatch: got %d, expected %d", input.Channels, conv.InChannels)
	}

	// Calculate output dimensions
	outHeight := (input.Height-conv.FilterSize+2*conv.Padding)/conv.Stride + 1
	outWidth := (input.Width-conv.FilterSize+2*conv.Padding)/conv.Stride + 1

	output := NewTensor3D(conv.NumFilters, outHeight, outWidth)

	// Perform convolution for each filter
	for f := 0; f < conv.NumFilters; f++ {
		for outH := 0; outH < outHeight; outH++ {
			for outW := 0; outW < outWidth; outW++ {
				sum := conv.Bias[f]

				// Convolve over all input channels
				for c := 0; c < conv.InChannels; c++ {
					for fh := 0; fh < conv.FilterSize; fh++ {
						for fw := 0; fw < conv.FilterSize; fw++ {
							inH := outH*conv.Stride + fh - conv.Padding
							inW := outW*conv.Stride + fw - conv.Padding

							// Check bounds
							if inH >= 0 && inH < input.Height && inW >= 0 && inW < input.Width {
								sum += input.Data[c][inH][inW] * conv.Filters[f][c][fh][fw]
							}
						}
					}
				}

				output.Data[f][outH][outW] = sum
			}
		}
	}

	return output, nil
}

// MaxPool2D performs 2D max pooling
type MaxPool2D struct {
	PoolSize int
	Stride   int
}

// NewMaxPool2D creates a new max pooling layer
func NewMaxPool2D(poolSize, stride int) *MaxPool2D {
	return &MaxPool2D{PoolSize: poolSize, Stride: stride}
}

// Forward performs max pooling
func (pool *MaxPool2D) Forward(input *Tensor3D) *Tensor3D {
	outHeight := (input.Height-pool.PoolSize)/pool.Stride + 1
	outWidth := (input.Width-pool.PoolSize)/pool.Stride + 1

	output := NewTensor3D(input.Channels, outHeight, outWidth)

	for c := 0; c < input.Channels; c++ {
		for outH := 0; outH < outHeight; outH++ {
			for outW := 0; outW < outWidth; outW++ {
				maxVal := input.Data[c][outH*pool.Stride][outW*pool.Stride]

				for ph := 0; ph < pool.PoolSize; ph++ {
					for pw := 0; pw < pool.PoolSize; pw++ {
						inH := outH*pool.Stride + ph
						inW := outW*pool.Stride + pw
						if input.Data[c][inH][inW] > maxVal {
							maxVal = input.Data[c][inH][inW]
						}
					}
				}

				output.Data[c][outH][outW] = maxVal
			}
		}
	}

	return output
}
