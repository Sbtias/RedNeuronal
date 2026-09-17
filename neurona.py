"""RDio Neural Core

Motor neuronal pequeño, sin dependencias externas. Incluye capas densas,
retropropagación, entrenamiento aleatorio, métricas y persistencia del modelo.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

Number = float | int


def sigmoid(x: Number) -> float:
    x = max(-40.0, min(40.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(value: float) -> float:
    return value * (1.0 - value)


def mse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if not predictions:
        return 0.0
    return sum((t - p) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


class NeuralNetwork:
    """Red neuronal densa sencilla.

    Ejemplo: 2 -> 8 -> 6 -> 1.
    Aprende patrones numéricos mediante ejemplos y puede guardar su estado.
    """

    def __init__(
        self,
        layers: Sequence[int] = (2, 8, 6, 1),
        learning_rate: float = 0.12,
        seed: int | None = None,
    ):
        if len(layers) < 2 or any(int(n) < 1 for n in layers):
            raise ValueError("La arquitectura debe tener al menos dos capas válidas.")
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que 0.")
        if seed is not None:
            random.seed(seed)

        self.layers = [int(n) for n in layers]
        self.learning_rate = float(learning_rate)
        self.epoch = 0
        self.loss = 1.0
        self._initialize()

    def _initialize(self) -> None:
        self.weights = []
        self.biases = []
        for input_size, output_size in zip(self.layers, self.layers[1:]):
            scale = math.sqrt(2.0 / input_size)
            self.weights.append([
                [random.uniform(-1.0, 1.0) * scale for _ in range(input_size)]
                for _ in range(output_size)
            ])
            self.biases.append([0.0] * output_size)

    def reset(self) -> None:
        self._initialize()
        self.epoch = 0
        self.loss = 1.0

    def forward(self, inputs: Sequence[Number], trace: bool = False):
        if len(inputs) != self.layers[0]:
            raise ValueError(
                f"Se esperaban {self.layers[0]} entradas, recibidas {len(inputs)}."
            )

        activation = [float(x) for x in inputs]
        activations = [activation[:]]

        for weights, biases in zip(self.weights, self.biases):
            activation = [
                sigmoid(bias + sum(weight * value for weight, value in zip(row, activation)))
                for row, bias in zip(weights, biases)
            ]
            activations.append(activation[:])

        return activations if trace else activation

    def predict(self, inputs: Sequence[Number]):
        return self.forward(inputs)

    def train(self, inputs: Sequence[Number], expected: Sequence[Number] | Number) -> float:
        """Entrena un ejemplo con retropropagación y devuelve su MSE."""
        activations = self.forward(inputs, trace=True)
        targets = [float(expected)] if isinstance(expected, (int, float)) else [float(x) for x in expected]

        if len(targets) != self.layers[-1]:
            raise ValueError(
                f"Se esperaban {self.layers[-1]} salidas objetivo, recibidas {len(targets)}."
            )

        deltas = [[] for _ in self.weights]
        output = activations[-1]
        deltas[-1] = [
            (target - value) * sigmoid_derivative(value)
            for value, target in zip(output, targets)
        ]

        for layer in range(len(self.weights) - 2, -1, -1):
            deltas[layer] = []
            for neuron_index, value in enumerate(activations[layer + 1]):
                error = sum(
                    self.weights[layer + 1][next_index][neuron_index]
                    * deltas[layer + 1][next_index]
                    for next_index in range(len(self.weights[layer + 1]))
                )
                deltas[layer].append(error * sigmoid_derivative(value))

        for layer, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            previous = activations[layer]
            for neuron_index, row in enumerate(weights):
                for input_index in range(len(row)):
                    row[input_index] += (
                        self.learning_rate
                        * deltas[layer][neuron_index]
                        * previous[input_index]
                    )
                biases[neuron_index] += self.learning_rate * deltas[layer][neuron_index]

        return mse(output, targets)

    def fit(
        self,
        data: Iterable[tuple[Sequence[Number], Sequence[Number] | Number]],
        epochs: int = 1000,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> list[float]:
        """Entrena todos los ejemplos y devuelve el historial de error."""
        samples = list(data)
        if not samples:
            raise ValueError("No hay datos para entrenar.")
        if epochs < 1:
            raise ValueError("epochs debe ser mayor que 0.")

        history = []
        for _ in range(epochs):
            batch = samples[:]
            if shuffle:
                random.shuffle(batch)

            total = sum(self.train(inputs, expected) for inputs, expected in batch)
            self.loss = total / len(batch)
            self.epoch += 1
            history.append(self.loss)

            if verbose and (self.epoch == 1 or self.epoch % max(1, epochs // 10) == 0):
                print(f"Época {self.epoch:>5} | error {self.loss:.6f}")

        return history

    def train_until(self, data, target_loss: float = 0.01, max_epochs: int = 10000) -> list[float]:
        """Entrena hasta alcanzar el error objetivo o el límite de épocas."""
        if target_loss <= 0:
            raise ValueError("target_loss debe ser mayor que 0.")

        history = []
        for _ in range(max_epochs):
            history.extend(self.fit(data, epochs=1, shuffle=True))
            if self.loss <= target_loss:
                break
        return history

    def serialize(self) -> dict:
        return {
            "layers": self.layers,
            "learning_rate": self.learning_rate,
            "epoch": self.epoch,
            "loss": self.loss,
            "weights": self.weights,
            "biases": self.biases,
        }

    def save(self, path: str | Path = "rdio_model.json") -> None:
        Path(path).write_text(
            json.dumps(self.serialize(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path = "rdio_model.json") -> "NeuralNetwork":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        brain = cls(data["layers"], data.get("learning_rate", 0.12))
        brain.weights = data["weights"]
        brain.biases = data["biases"]
        brain.epoch = int(data.get("epoch", 0))
        brain.loss = float(data.get("loss", 1.0))
        return brain


if __name__ == "__main__":
    # XOR comprueba que la red pueda aprender un patrón no lineal.
    data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    brain = NeuralNetwork(layers=(2, 8, 6, 1), learning_rate=0.12, seed=42)
    brain.train_until(data, target_loss=0.01, max_epochs=10000)

    print("\nRDio Neural Core")
    print(f"Arquitectura: {' -> '.join(map(str, brain.layers))}")
    print(f"Épocas: {brain.epoch}")
    print(f"Error: {brain.loss:.6f}\n")

    for inputs, expected in data:
        prediction = brain.predict(inputs)[0]
        print(f"{list(inputs)} -> {prediction:.4f} (esperado: {expected[0]})")

    brain.save("rdio_model.json")
