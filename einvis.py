import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

def display_einsum_inputs(tensor_a, tensor_b):
    # Ensure tensors are numpy arrays
    tensor_a = np.array(tensor_a)
    tensor_b = np.array(tensor_b)
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(12, 5))
    
    # Display tensor A
    ax1 = fig.add_subplot(121, projection='3d')
    display_tensor_in_axis(ax1, tensor_a, 'Tensor A', color='red')
    
    # Display tensor B
    ax2 = fig.add_subplot(122, projection='3d')
    display_tensor_in_axis(ax2, tensor_b, 'Tensor B', color='blue')
    
    plt.tight_layout()
    plt.show()

def apply_rotation(tensor, einsum_notation):
    input_subscripts, output_subscript = einsum_notation.split('->')
    a_subscript, b_subscript = input_subscripts.split(',')
    
    # Determine the new order of axes
    new_order = []
    for i, dim in enumerate(a_subscript):
        if dim not in b_subscript:
            tensor = np.expand_dims(tensor, axis=-1)
            new_order.append(tensor.ndim - 1)
        else:
            new_order.append(b_subscript.index(dim))
    
    for i, dim in enumerate(b_subscript):
        if i not in new_order:
            new_order.append(i)
    
    # Apply the rotation
    return np.transpose(tensor, new_order)

def display_tensor_in_axis(ax, tensor, title, color):
    # Pad tensor to 3D if necessary
    shape = list(tensor.shape)
    max_dim = max(shape + [1])  # Ensure at least 1D
    padded_shape = [max_dim] * 3
    padded_tensor = np.zeros(padded_shape)
    
    # Copy original tensor data into padded tensor
    if tensor.ndim == 0:
        padded_tensor[0, 0, 0] = tensor
    elif tensor.ndim == 1:
        padded_tensor[:tensor.shape[0], 0, 0] = tensor
    elif tensor.ndim == 2:
        padded_tensor[:tensor.shape[0], :tensor.shape[1], 0] = tensor
    elif tensor.ndim == 3:
        padded_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor

    # Normalize tensor values to [0, 1] range for color mapping
    normalized_tensor = (padded_tensor - np.min(padded_tensor)) / (np.max(padded_tensor) - np.min(padded_tensor) + 1e-8)

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("custom", [(1,1,1,0), color])

    # Plot voxels
    ax.voxels(padded_tensor != 0, facecolors=cmap(normalized_tensor), edgecolors='k')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\nShape: {tensor.shape}')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def display_aligned_einsum_inputs(tensor_a, tensor_b, einsum_notation):
    rotated_tensor_b = apply_rotation(tensor_b, einsum_notation)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    display_tensor_in_axis(ax1, tensor_a, 'Tensor A', color='red')
    
    ax2 = fig.add_subplot(122, projection='3d')
    display_tensor_in_axis(ax2, rotated_tensor_b, 'Rotated Tensor B', color='blue')
    
    plt.tight_layout()
    plt.show()

def replicate_tensors(a, b):
    max_shape = np.maximum(a.shape, b.shape)
    replicated_a = np.broadcast_to(a, max_shape)
    replicated_b = np.broadcast_to(b, max_shape)
    return replicated_a, replicated_b

def display_replicated_einsum_inputs(a, b, einsum_notation):
    # Rotate tensor B first
    rotated_b = apply_rotation(b, einsum_notation)

    # Add dims to the end of a until it has as many as rotated_b
    while a.ndim < rotated_b.ndim:
        a = np.expand_dims(a, axis=-1)
    
    # Replicate tensors
    replicated_a, replicated_b = replicate_tensors(a, rotated_b)
    
    # Display replicated tensors
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    display_tensor_in_axis(ax1, replicated_a, 'Replicated Tensor A', color='red')
    
    ax2 = fig.add_subplot(122, projection='3d')
    display_tensor_in_axis(ax2, replicated_b, 'Rotated and Replicated Tensor B', color='blue')
    
    plt.tight_layout()
    plt.show()

def display_multiplication_tensor(a, b, einsum_notation):
    # Rotate tensor B first
    rotated_b = apply_rotation(b, einsum_notation)

    # Add dims to the end of a until it has as many as rotated_b
    while a.ndim < rotated_b.ndim:
        a = np.expand_dims(a, axis=-1)
    
    # Replicate tensors
    replicated_a, replicated_b = replicate_tensors(a, rotated_b)
    
    # Perform element-wise multiplication
    multiplication_tensor = replicated_a * replicated_b
    
    # Display multiplication tensor
    fig = plt.figure(figsize=(6, 5))
    
    ax = fig.add_subplot(111, projection='3d')
    display_tensor_in_axis(ax, multiplication_tensor, 'Multiplication Tensor', color='purple')
    
    plt.tight_layout()
    plt.show()

def display_final_tensor(a, b, einsum_notation):
    # Perform the einsum operation
    final_tensor = np.einsum(einsum_notation, a, b)
    
    # Display final tensor
    fig = plt.figure(figsize=(6, 5))
    
    ax = fig.add_subplot(111, projection='3d')
    display_tensor_in_axis(ax, final_tensor, 'Final Tensor', color='purple')
    
    plt.tight_layout()
    plt.show()

def validate_einsum(tensor_a, tensor_b, einsum_notation):
    input_subscripts, output_subscript = einsum_notation.split('->')
    a_subscript, b_subscript = input_subscripts.split(',')
    
    if len(a_subscript) != tensor_a.ndim or len(b_subscript) != tensor_b.ndim:
        raise ValueError("Einsum notation doesn't match tensor dimensions")
    
    if len(output_subscript) > 3:
        raise ValueError("Output tensor must have 3 or fewer dimensions for visualization")
    
    if tensor_a.ndim > 3 or tensor_b.ndim > 3:
        raise ValueError("Input tensors must have 3 or fewer dimensions for visualization")

def visualize_einsum(tensor_a, tensor_b, einsum_notation):
    validate_einsum(tensor_a, tensor_b, einsum_notation)
    
    display_einsum_inputs(tensor_a, tensor_b)
    display_aligned_einsum_inputs(tensor_a, tensor_b, einsum_notation)
    display_replicated_einsum_inputs(tensor_a, tensor_b, einsum_notation)
    display_multiplication_tensor(tensor_a, tensor_b, einsum_notation)
    display_final_tensor(tensor_a, tensor_b, einsum_notation)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize einsum operation')
    parser.add_argument('--a', type=str, default='[[1, 2], [3, 4]]', help='Tensor A as a string representation of a list')
    parser.add_argument('--b', type=str, default='[[1, 1], [2, 2], [3, 3]]', help='Tensor B as a string representation of a list')
    parser.add_argument('--einsum', type=str, default='ab,cb->ac', help='Einsum notation')
    
    args = parser.parse_args()
    
    tensor_a = np.array(eval(args.a))
    tensor_b = np.array(eval(args.b))
    einsum_notation = args.einsum
    
    visualize_einsum(tensor_a, tensor_b, einsum_notation)