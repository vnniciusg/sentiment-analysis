from functools import wraps
from time import perf_counter
from typing import Callable

import torch
from loguru import logger
from matplotlib import pyplot as plt


def timeit(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {perf_counter() - start:.2f} seconds")
        return result

    return wrapper


def exception_handler(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"an exception in {func.__name__} ocurred: {str(e)}")

    return wrapper


def debugit(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"calling {func.__name__} with arguments: {args}, {kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} returned: {result}")
        return result

    return wrapper


def vizualize_attetion(tokens: list, attn_weights: torch.Tensor) -> None:
    weights = attn_weights.squeeze().detach().numpy()[: len(tokens)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(weights)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title("attention weights per token")
    plt.show()
