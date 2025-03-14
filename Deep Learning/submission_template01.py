import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256),  # Первый линейный слой (784 -> 256)
        nn.ReLU(),            # Функция активации для скрытого слоя
        nn.Linear(256, 16),   # Второй линейный слой (256 -> 16)
        nn.ReLU(),            # Функция активации для скрытого слоя
        nn.Linear(16, 10)     # Последний линейный слой (16 -> 10)
    )
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Пример с маленькой моделью
small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Что-то не так, количество параметров неверное'

# Пример с моделью на основе Sequential
medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Что-то не так, количество параметров неверное'

print("Seems fine!")
