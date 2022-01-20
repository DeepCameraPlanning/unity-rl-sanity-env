# Getting Started

## Unity

Version: `2020.3.24f1`

1. Open the project with Unity
2. Install `2.0.1 ML-Agents` package (via the Package Manager window)
3. Build the project at the repository root, with the name: `sanity-env-discrete` (check the `Run In Background` option in `Project Settings`)

## Python

Version: `3.7.11`

1. Run `pip install -r requirements.txt`
2. Run `python Python/run.py` to launch the training

Note: if you have local file import troubles, check your `PYTHONPATH`.

## Usage

Command to train (with a well set up config):
`python Python/run.py`

Command to infer:
`python Python/run.py run_type=infer model.checkpoint_path=path/to/checkoint.pt`