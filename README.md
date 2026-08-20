# MIMO_sim

MIMO channel simulation scripts for indoor (InH) and indoor-factory (InF) scenarios, including 3GPP-style channel model generation, beam allocation, and channel capacity (water-filling) calculation. Some scripts support GPU-accelerated computation via PyTorch.

## Key modules

- `Channel_functions.py` — core channel model generation (path loss, angular spread, clusters, etc.)
- `Channel_function_gpu.py` — GPU-accelerated version of the channel model
- `ChannelMatrix_Calculation.py` — MIMO channel matrix construction
- `Beam_Allocation.py` — beam allocation logic
- `DrawGraph.py` — plotting/visualization utilities

## Requirements

Python 3.13, with `numpy`, `scipy`, `matplotlib`, `numba`, `opt_einsum`, `pandas`, and `torch` (for GPU scripts).
