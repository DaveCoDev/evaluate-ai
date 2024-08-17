# evaluate-ai
evaluate-ai is a tool for easily developing custom evaluations for language models.

Note that this repo is experimental and features may break with updates.

| Problem | evaluate-ai |
| --- | --- |
| Understanding which model is capable enough for **your** specific use cases. Open-source benchmarks are useful for understanding general capabilities, but there may not be one that transfers well to your task. | Provides a common interface to define new custom evaluations and log the results of executing them. |
| Open-source benchmarks may end up leaking into the training set of the model. | Create custom evaluations that can be kept private. |
| Many aspects of models we wish to evaluate (where there exists some ground truth or criteria to validate model outputs against) fall into a common pattern that can be parameterized and thus we want to create them easily with no-code | Provides common patterns out of the box. For example, [EvaluationContainsPattern](./evaluate_ai/evaluations/contains_pattern.py) checks if the model output contains a provided regex. | 
| Monitoring changes in performance over time, particularly for web-app based providers like ChatGPT or Microsoft Copilot that abstract the model and algorithms they use. | Provides a terminal program to provide results from a web interface and run evaluations on that output. |


## Usage
evaluate-ai uses [Poetry](https://python-poetry.org/docs/) for managing the Python environment and dependencies.

This repo is best used by adding to and modifying it directly. As such, it is recommended to clone or create a fork of the repository. 
After doing so, run the following command to create a Python environment, if it does not exist, and install its dependencies:
```bash
poetry install
```

Alternatively, you can pip install from the remote with the following command, although this is not recommended.
```bash
pip install evaluate-ai @ git+https://github.com/DaveCoDev/evaluate-ai
```


### Scripts
Once the package is installed with Poetry, you can run several scripts via terminal commands. See the [pyproject.toml](pyproject.toml) file for the entry point of each command.

#### Run Evaluations
These evaluations require a language model service configured - see the below section for more details on configuring models. Once configured, define the model name(s) to use in config.yaml.

By default, this will execute all the evaluations as defined in [data/evaluations/](./data/evaluations/) and [config.yaml](./data/evaluations/config.yaml). See the section below on configuring new instances of existing evaluations with no-code. The results of each evaluation instance will automatically be saved to a tinydb database in the [data](./data/) directory.

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

**Important: The combination of  `name` and `type` fields must be unique.**

The general structure of the files is that each file is a list of items where each item corresponds to one instance of an evaluation. 
The required fields are:
- `name`: A friendly name for the evaluation instance used to identify it in the results.
- `type`: The evaluation that will executed is determined by the `type` field, which must be a key in the [evaluation registry](./evaluate_ai/evaluation_registry.py). For example, `contains_pattern`.
- `parameters`: Any additional parameters that the evaluation requires are passed as kwargs to the evaluation class. The required parameters are defined in the corresponding evaluation class.

Global configuration options, such as selecting which models **all** evaluations should be evaluated against, can be set in the [config.yaml](./data/evaluations/config.yaml) file.
The current configuration options are:
- `models`: A list of models to evaluate against by provider. The model names must be supported by the `Evaluation.call_llm`.
- `evaluate_external` Options for running manual evaluations
  - `models`: A list of providers to evaluate against. For example `copilot-creative` may refer to Microsoft Copilot's creative mode.
  - `evaluation_names`: Since it would be tedious to run all evaluations as the `run-evaluations` script does, explicitly list the evaluations to run manually here.


### Configuring Models
We currently use [not-again-ai](https://github.com/DaveCoDev/not-again-ai/tree/main) as the API to interact with multiple language models. 
Its [common `chat_completion`](https://github.com/DaveCoDev/not-again-ai/blob/main/notebooks/llm/common_chat_completion.ipynb) interface currently supports [OpenAI](https://platform.openai.com/docs/models) and [Ollama](https://ollama.com/library?sort=popular) models. Check [not-again-ai's documentation](https://github.com/DaveCoDev/not-again-ai?tab=readme-ov-file#installation) for the most up-to-date setup instructions. Roughly, the steps will involve either setting environment variables or installing Ollama for your system.

To use other models or otherwise modify how language model responses are generated, the [Evaluation.call_llm](./evaluate_ai/evaluation.py) method can be modified.


### Adding a New Evaluation
1. **Create a new class that is a subclass of `Evaluation`.**
    - Example evaluation classes are under [evaluate_ai/evaluations/](./evaluate_ai/evaluations/). 
    - [EvaluationContainsPattern](./evaluate_ai/evaluations/contains_pattern.py) is a good example to start with.
    - The class must implement the following:
        - Set `self.evaluation_data.name` to be some friendly identifier for the evaluation. Usually this is provided from the config (see below step on create a config file).
        - Set `self.evaluation_data.type` to be the name of the evaluation in the evaluation registry. See the below step for more information on the registry.
        - Implement the `get_result` method. This method should set `self.evaluation_data.metadata.output` to the output(s) of the model which can then be used in the `evaluate` method. Note that `call_llm` automatically sets the model output.
        - Implement the `evaluate` method. This must set `self.evaluation_data.score` which is the score from 0-100 of the evaluation.
        - (Optional) Implement the `task_as_string` method. This will be used to display the task in a friendly way which is useful for manual evaluation.
2. **Add the evaluation type to `EVALUATION_REGISTRY` at [evaluate_ai/evaluation_registry.py](./evaluate_ai/evaluation_registry.py)**
    - The key must be the `self.evaluation_data.type`.
    - The value must be the name of the class.
3. **Create a config file`.yaml` under [data/evaluations/](./data/evaluations/)**
    - The config file must have the following fields:
        - `name`: A friendly name for the evaluation.
        - `type`: The key that was set in the aforementioned evaluation registry.
      - Other fields must be under `parameters` and will be passed as kwargs to the evaluation class. Everything else will be ignored.


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