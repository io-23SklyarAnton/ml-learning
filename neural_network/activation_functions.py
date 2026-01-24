import enum


class ActivationFunction(str, enum.Enum):
    RELU = "ReLU"
    SIGMOID = "sigmoid"
