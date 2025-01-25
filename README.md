# evaluate-ai
evaluate-ai is a tool for easily developing custom evaluations for language models.

Note that this repo is experimental and features may break with updates.

| Problem | evaluate-ai |
| --- | --- |
| Understanding which model is capable enough for **your** specific use cases. Open-source benchmarks are useful for understanding general capabilities, but there may not be one that transfers well to your task. | Provides a common interface to define new custom evaluations and log the results of executing them. |
| Open-source benchmarks may end up leaking into the training set of the model. | Create custom evaluations that can be kept private. |
| Many aspects of models we wish to evaluate (where there exists some ground truth or criteria to validate model outputs against) fall into a common pattern that can be parameterized and thus we want to create them easily with no-code | Provides common patterns out of the box. For example, [EvaluationContainsPattern](./evaluate_ai/evaluations/contains_pattern.py) checks if the model output contains a provided regex. | 


## Usage
evaluate-ai uses [Poetry](https://python-poetry.org/docs/) for managing the Python environment and dependencies.

This repo is best used by adding to and modifying it directly. As such, it is recommended to clone or create a fork of the repository. 
After doing so, run the following command to create a Python environment, if it does not exist, and install its dependencies:
```bash
poetry sync
```

Alternatively, you can pip install from the remote with the following command, although this is not recommended.
```bash
pip install evaluate-ai @ git+https://github.com/DaveCoDev/evaluate-ai
```


### Scripts
Once the package is installed with Poetry, you can run several scripts via terminal commands. See the [pyproject.toml](pyproject.toml) file for the entry point of each command.

#### Run Evaluations
These evaluations require a language model service configured - see the below section for more details on configuring models. Once configured, define the model name(s) to use in config.yaml.

By default, this will execute all the evaluations as defined in [data/evaluations/](./data/evaluations/). See the section below on configuring new instances of existing evaluations with no-code. The results of each evaluation instance will automatically be saved to a tinydb database in the [data](./data/) directory.

```bash
run-evaluations
```

#### View Results
Prints the latest (by timestamp of last execution time) results for each evaluation and each model pair to console.

```bash
view-results
```


### Configuring New Instances of Existing Evaluations with No-Code
For each evaluation implemented by default, there is a corresponding `yaml` file that shows examples of the parameters available for the evaluation.
The parameters for each evaluation are defined in a `EvaluationInstance` class under the responding [evaluation module](./evaluate_ai/evaluations/).


### Configuring Models
We currently use [not-again-ai](https://github.com/DaveCoDev/not-again-ai/tree/main) as the API to interact with multiple language models. However each evaluation can be changed to use any model service.
The clients for various LLM providers are defined in [constants.py](./evaluate_ai/constants.py). This is where you can configure things like API keys.
Check [not-again-ai's documentation](https://github.com/DaveCoDev/not-again-ai?tab=readme-ov-file#installation) for the most up-to-date setup instructions. 


### Adding a New Evaluation
The easiest way to add a new evaluation is to follow the structure of the existing evaluations, such as [ContainsPattern](./evaluate_ai/evaluations/contains_pattern.py) and [01_contains_pattern.yaml](./data/evaluations/01_contains_pattern.yaml). 
The Python module can extend and override any of the classes in [evaluate_ai/evaluation.py](./evaluate_ai/evaluation.py) to define custom behavior.


## Roadmap
- [ ] **Additional Evaluations**
    - [ ] (Python) Coding Problems. Input a natural language description of a problem, optional context, and test cases. 
    The model should output Python which will be executed and tested against the test cases.
- [ ] **Better Visualization and Reporting of Results**


## Misc
### Typos
Check for typos using [typos](https://github.com/crate-ci/typos)

```bash
typos -c ./.github/_typos.toml
```