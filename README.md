# Einsum Visualizer

This project provides a visualization tool for Einstein summation (einsum) operations. It renders the input tensors, their alignment, multiplication, and the resulting output as 3D cuboids.

## Features

- Supports einsums with input tensors up to 3 combined named dimensions
- Visualizes output tensors up to 3 dimensions
- Renders tensors as n-dimensional cuboids with color-coded elements
- Shows step-by-step visualization of the einsum operation

## Usage

To run the einsum visualizer, use the following command:

```
python einvis.py [--a A] [--b B] [--einsum EINSUM]
```

### Example

```
python einvis.py --a "[[1, 2], [3, 4]]" --b "[[1, 1], [2, 2], [3, 3]]" --einsum "ab,cb->ac"
```

This command will visualize the einsum operation `ab,cb->ac` with the given input tensors.

## Requirements

- Python 3.x
- Required Python packages (list them here, e.g., numpy, matplotlib, etc.)

## License

Do what you like