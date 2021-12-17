# SportsSchedulingRLGNN
## Main Workflow
1. `GraphXMLGeneration.py` creates synthetic instances with a predefined # of teams and a guaranteed solution.
1. `Graph.py` generates Graph objects from instance `xml` files and saves them as pickles.
2. `training.py` trains `RLAgent.py` over the preprocessed instances from 1.
    - Hyperparams are stored in `training.py`
    - `RLAgent.py` is paramaterized by either `s2v_scheduling.py` or `s2v_schedulingNew.py`
3. `RLAgent.py` is evaluated against `GreedyAgent.py` and `RandomAgent.py` using the script in `evaluation.py`

## Secondary Files
- `GraphSummary.py` generates a CSV summary of Graph instances
- `NodeAndEdge.py` a supporting class for `Graph.py`
- `tablesandfigures.py` generates training curves
- `GraphVis.py` creates a visual representation of a given Graph.
