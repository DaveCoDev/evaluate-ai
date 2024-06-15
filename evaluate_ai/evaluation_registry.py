from enum import Enum

EVALUATION_REGISTRY = {
    "contains_pattern": "EvaluationContainsPattern",
    "structured_output": "EvaluationStructuredOutput",
}

EVALUATION_ENUM = Enum("EvaluationEnum", list(EVALUATION_REGISTRY.keys()))
