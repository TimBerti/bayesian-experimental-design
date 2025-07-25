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
         {"name": "param2", "type": "int", "bounds": [1, 10]},
         {"name": "param3", "type": "choice", "values": [1, 2, 4], "is_ordered": true}
         {"name": "param4", "type": "choice", "values": ["red", "green"]},
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
          {"name": "param2", "type": "int", "bounds": [1, 10]},
          {"name": "param3", "type": "choice", "values": [1, 2, 4], "is_ordered": true},
          {"name": "param4", "type": "choice", "values": ["red", "green"]}
      ],
      "samples": [
          {"param1": 0.123, "param2": 5, "param3": 2, "param4": "red", "result": null},
          {"param1": 0.456, "param2": 8, "param3": 4, "param4": "green", "result": null},
          {"param1": 0.789, "param2": 3, "param3": 1, "param4": "red", "result": null},
          {"param1": 0.234, "param2": 6, "param3": 2, "param4": "green", "result": null},
          {"param1": 0.567, "param2": 9, "param3": 4, "param4": "red", "result": null}
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
| Ordered      | `{"type": "choice", "values": [1, 2, 4], "is_ordered": true}` | GP exploits ordering  |
| Categorical  | `{"type": "choice", "values": ["red", "green"]}`              | One‑hot encoded       |

---

### Command‑line flags

| Flag(s)        | Meaning                                                              | Default |
| -------------- | -------------------------------------------------------------------- | ------- |
| `file`         | Path to the JSON experiment file                                     | –       |
| `--samples`    | Number of new samples to propose                                     | `1`     |
| `--noise`      | Noise level for the Gaussian Process (in the same units as `result`) | `1e-6`  |
| `--seed`       | RNG seed for reproducibility                                         | `None`  |