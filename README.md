# Bayesian Experimental Design

Batched Bayesian experimental design via **JSON I/O** built on **BoTorch** with information gain acquisition.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

1. **Create a minimal `experiment.json`** (samples can be empty):

   ```json
   {
     "parameters": [
         {"name": "param1", "type": "range", "bounds": [0.0, 1.0]},
         {"name": "param2", "type": "choice", "values": ["red", "green"]},
         {"name": "param3", "type": "choice", "values": [1, 2, 4], "is_ordered": true}
     ],
     "samples": []
   }
   ```
2. **Ask for an initial batch** (5 Sobol points):

   ```bash
   python doe.py experiment.json --samples 5 --seed 42
   ```

   Four new entries (missing `result`) are appended.
   ```json
   {
     "parameters": [
         {"name": "param1", "type": "range", "bounds": [0.0, 1.0]},
         {"name": "param2", "type": "choice", "values": ["red", "green"]},
         {"name": "param3", "type": "choice", "values": [1, 2, 4], "is_ordered": true}
     ],
     "samples": [
         {"param1": 0.123, "param2": "red", "param3": 1, "result": null},
         {"param1": 0.456, "param2": "green", "param3": 2, "result": null},
         {"param1": 0.789, "param2": "red", "param3": 4, "result": null},
         {"param1": 0.234, "param2": "green", "param3": 1, "result": null},
         {"param1": 0.567, "param2": "red", "param3": 2, "result": null}
     ]
   }
   ```
3. **Run your experiment**, fill each new entry’s `result`, and **iterate**:

   ```bash
   python doe.py experiment.json --samples 5 --seed 42
   ```

---

## Parameter Types

| type         | JSON schema                                                   | Notes                 |
| ------------ | ------------------------------------------------------------- | --------------------- |
| Continuous   | `{"type": "range", "bounds": [lo, hi]}`                       | `float` bounds        |
| Integer      | `{"type": "int", "bounds": [lo, hi]}`                         | `int` bounds          |
| Discrete‑num | `{"type": "choice", "values": [1, 2, 4], "is_ordered": true}` | GP exploits ordering  |
| Categorical  | `{"type": "choice", "values": ["red", "green"]}`              | One‑hot encoded       |

---

### Command‑line flags

| Flag(s)        | Meaning                               | Default |
| -------------- | ------------------------------------- | ------- |
| `file`         | Path to the JSON experiment file      | –       |
| `--samples`    | Number of new samples to propose      | `1`     |
| `--noise`      | Noise level for the Gaussian Process  | `1e-6`  |
| `--seed`       | RNG seed for reproducibility          | `None`  |