# Tree Source Localization

A Python package for modeling infection source localization on tree graphs with probabilistic edge delay distributions using Moment Generating Functions.

---
## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
  - [Inputs](#input-json-file-format)
  - [Usage](#key-methods-documentation-and-examples)
- [Changelog](#changelog)

---
## Installation
To install directly:
   ```
   pip install git+https://github.com/Decos14/Tree_Source_Localization.git
   ```
Or clone the repository locally and install:

   ```
   git clone https://github.com/yourusername/Tree_Source_Localization.git
   cd Tree_Source_Localization
   pip install .
   ```
---
## Documentation
### Input JSON File Format
The input format is a JSON file with the following format:
```
{"node_name,node_name": {distribution: 'distribution type', parameters : {'paramater1': value1, ...} } }
```
note that the order of the node names is irrelevant to the parsing of the tree.

Distribution type codes:

- 'N': Positive Normal  
- 'E': Exponential  
- 'U': Uniform  
- 'P': Poisson  
- 'C': Absolute Cauchy

Example:
```
{
    "A,B" : {
        'distribution's : 'N',
        'parameters' : {
            'mu': 3.0,
            'sigma2': 1.0
        }
  },
    "A,C" : {
        'distribution' : 'E',
        'parameters' : {
            'lambda': 0.5
        }
    },
    "C,D" : {
        'distribution' : 'U',
        'paramaters' : {
            'start' : 0.0,
            'stop' : 1.0
        }
    }
}
```
---

### Key Methods Documentation and Examples

#### `build_tree(file_name: str) -> None`

Builds the tree data structure from a JSON file, parsing edges, nodes, distributions, parameters, delays, and MGF functions.

```
tree.build_tree("tree_topology.json")
```

---

#### `simulate() -> None`

Simulates delay values for all edges using their respective distributions and updates `self.edge_delays`.

```
tree.simulate()
print(tree.edge_delays)  # Access simulated delays for each edge
```

---

#### `simulate_infection(source: str) -> None`

Simulates infection spread times from a given source node to all observers, storing results in `self.infection_times`.

```
tree.simulate_infection("nodeA")
print(tree.infection_times)  # Infection times per observer from source "nodeA"
```

---

#### `joint_mgf(u: ArrayLike, source: str) -> float`

Computes the joint Moment Generating Function (MGF) of infection times for observers given a source node, evaluated at vector `u`.

```
import numpy as np
u = np.array([1.0, 0.5, 0.3])
value = tree.joint_mgf(u, "nodeA")
print(value)
```

---

#### `cond_joint_mgf(u: ArrayLike, source: str, obs_o: str, method: str) -> float`

Computes or approximates the conditional joint MGF of observers given the first infected observer `obs_o` using a specified augmentation method.

- `method` options:  
  'linear' = Linear approximation  
  'exponential' = Exponential approximation  
  'exact' = Exact solution (iid exponential delays)

```
value = tree.cond_joint_mgf(u, "nodeA", "observer1", method=1)
print(value)
```

---

#### `get_equivalent_class(first_obs: str, outfile: str) -> List[str]`

Computes the equivalence class of nodes sufficient for source estimation based on the first infected observer `first_obs`, writes the relevant subtree edges to `outfile`, and returns relevant observers.

```
relevant_observers = tree.get_equivalent_class("observer1", "subtree.json")
print(relevant_observers)
```

---

#### `obj_func(u: ArrayLike, source: str, method: str = None) -> float`

Objective function used to identify the most likely infection source. Accepts optional augmentation.

```
val = tree.obj_func(u, "nodeA", augment='exponential')
print(val)
```

---

#### `localize(method: str = None) -> str`

Estimates the most likely infection source node by minimizing the objective function, optionally using augmentation.

```
predicted_source = tree.localize(method='linear')
print(f"Predicted source: {predicted_source}")
```

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.