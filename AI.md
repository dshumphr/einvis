An einsum visualizer.

Supports einsums where the input tensors have up to 3 named dimensions combined and output tensors are up to 3 dimensions.

Renders the tensors as n-dimensional cuboids with different colors, where the opaqueness of a sub-cube (representing a single element) is scaled by the element's value.

The visualization shows the following:
Input tensor A, input tensor B.
A and B rotated so that dimensions are aligned as specified by the einsum.
A and B with replication along any axis where it is required by the einsum.
A multiplication tensor that represents multiplying all the aligned elements.
A final output tensor after summation along some dimensions has occured, if specified by the einsum.