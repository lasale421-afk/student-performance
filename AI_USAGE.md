# AI Usage Declaration

**Project:** Student Risk Prediction  
**Authors:** Mohamed Reda Zarrouk and Georgy Salameh  
**AI assistant:** Claude Code (Claude Opus 5.5, Anthropic)

## 1. Overview

AI assistance was used throughout the project, mainly to support implementation, testing, documentation, and the analysis of results. The project direction was not delegated entirely to the assistant: we defined the problem and objectives, selected the dataset, specified the modelling approach and evaluation requirements, and guided the assistant's work according to those decisions.

We reviewed the generated code and documentation, ran the tests and project commands, checked results against the data where possible, and used explanations of the implementation to understand how the project works. Our aim was to use AI as a development assistant while retaining responsibility for the project's design, validation, and final decisions.

## 2. Our decisions and contribution

The following elements were defined or directed by us:

- The problem to investigate, the project objectives, and the dataset to use.
- The target definition and the rule to exclude `G1` and `G2` to avoid target leakage.
- The choice of models to compare and the evaluation priorities, including prioritising recall and setting a target recall of at least 0.80.
- The model-comparison procedure, cross-validation requirements, hyperparameter grids, model-selection rule, and threshold-selection criteria.
- The requirements for the command-line interface, prediction workflow, tests, and analysis notebook.
- Review of the assistant's output, requests for corrections or additional checks, and decisions about which changes to keep.

AI assistance was used to help clean and prepare the data, implement the Python code, create tests, document the project, and examine the results. For the implementation, we provided requirements and constraints intended to keep the code coherent with the project design. We also asked for independent checks of parts of the model comparison and analysis.

## 3. Use of AI during development

### Data preparation and baseline

The assistant helped implement the project structure, configuration, dataset loading and preparation, feature preprocessing, baseline models, and initial tests. We specified the dataset, target, leakage rule, Python requirements, and evaluation priorities.

We then asked the assistant to check factual statements against the downloaded data. During this review, two statements were corrected: the default model configuration did not produce a `ConvergenceWarning` on our data, and the preprocessing produced 56 one-hot encoded columns. The remaining statements that had not been independently verified were marked in the documentation. A citation to Cortez and Silva (2008) was also added.

### Model comparison and threshold selection

The assistant helped implement the model-training and evaluation workflow, including cross-validation, hyperparameter search, threshold selection, and reports. We specified the models to compare, the evaluation metric used for model selection, the one-standard-deviation selection rule, and the threshold objective: achieve recall of at least 0.80 and then favour the best precision among eligible thresholds.

After reviewing the initial results, we requested wider hyperparameter grids and a threshold trade-off report for recall targets of 0.70, 0.80, and 0.90. The comparison was rerun and the associated reports and documentation were updated. We also requested tests for invalid threshold inputs.

### Prediction interface, tests, and documentation

The assistant helped implement model saving and loading, prediction for a new case, command-line commands (`train`, `evaluate`, and `predict`), and the README. It also generated and expanded automated tests and wrote explanations of the source files and tests.

We specified the expected behaviour and asked for tests covering invalid inputs and common failure cases. We ran the tests and project commands, including a fresh-environment check, rather than relying only on the assistant's description of what the code should do.

### Analysis notebook and figures

The analysis notebook was developed collaboratively with AI assistance. The assistant helped organise the notebook, add sensitivity and group-level analyses, and create figures for the precision-recall curve, confusion matrix, and model coefficients.

A constraint we set was that the test set had already been used for evaluation and should not be used to make further modelling changes. After the notebook was run, we checked that the saved report files and test-set results had not changed. The notebook's sensitivity and group-level results were based on the training data and out-of-fold predictions as documented in the project.

## 4. Validation and checks

We did not treat generated output as automatically correct. We used tests, execution, and targeted reviews to check the implementation and its results.

The checks documented during development included:

- Automated tests for data preparation, features, evaluation, model training, saving and loading, and the command-line interface.
- Verification of selected dataset facts against the downloaded data and the dataset reference.
- Tests for invalid inputs and expected error behaviour.
- Fresh-environment checks on Python 3.10.19 and Python 3.12.3, following the installation and usage steps in the README.
- Re-running training, evaluation, prediction, and notebook execution to check reproducibility.
- Comparing report files and evaluation results before and after notebook analysis to confirm that the model and threshold had not been changed.

At the final documented check, 19 automated tests passed. In the fresh environments, the documented training, evaluation, and prediction commands succeeded. The held-out test results recorded during that check were ROC-AUC 0.778 and average precision 0.346. At a threshold of 0.140, precision was 0.300, recall was 0.750, and F1 was 0.429. These figures are reported as evaluation results, not as evidence that the model meets the target recall of 0.80 on the test set.

## 5. Remaining verification

Some explanatory statements were based on the assistant's prior knowledge rather than directly confirmed by the downloaded data or the course material. These were marked in the documentation with **[from prior knowledge, to verify]** so that they could be checked before the project defence.

To locate these statements in the documentation, run:

```bash
grep -rn "prior knowledge" docs/
```

## 6. Summary

AI was a substantial part of the implementation and documentation process. We used it to accelerate coding, testing, and analysis, while providing the project objectives, dataset choice, modelling requirements, evaluation rules, and implementation constraints. We reviewed the output, requested corrections and checks, executed the project, and examined the resulting code and explanations so that we could understand and justify the work.

We remain responsible for the final project and for being able to explain its design, implementation, limitations, and results.
