# AlphaTSP: Learning a TSP Heuristic Using the AlphaZero Methodology
Felix Parker and Darius Irani

[AlphaTSP](https://github.com/flixpar/AlphaTSP/blob/master/AlphaTSP.pdf)

Run with:

```bash
python3 main.py --experiment <experiment_name>
```
Where experiment name is one of:
- nearestneighbor
- mcts
- exact
- gurobi
- insertion
- policy
- parallel
- selfplay

selfplay is the primary experiment which implements the AlphaTSP method

Python packages required to run this project are listed in requirements.txt