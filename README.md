# SymPyToTorch
Create computation graphs from SymPy functions.

I needed to evaluate the same SymPy function over and over again. This was a lot slower than I had expected. Adding to that, the rest of the code was in Torch and could be run on CUDA. And so this was born. Given a SymPy method, it would return another method, which can be called using the inputs to the computation graph as arguments, and which returns the computation graph. It can then be easily run on CUDA or integrated with existing Torch code.

This is a WIP and only the functions that I required have been added in the format I needed. However, extending this to newer functions is extremely simple.
