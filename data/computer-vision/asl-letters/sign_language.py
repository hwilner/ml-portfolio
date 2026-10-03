"""Loader for the ASL letter image set.

The original repository carried a `sign_language.py` inside a `datasets/`
folder that was not a Python package, and the notebook imported it with
`from datasets import sign_language` - which fails, both because the folder has
no `__init__.py` and because the unrelated `datasets` distribution on PyPI
shadows the name. It also used `keras.utils.np_utils`, removed in Keras 3.

This version is self-contained and depends only on NumPy, Pillow and the
standard library, so it works under any Keras version (or none).

Example:
    Typical use, once the photographs are present under
    ``data/computer-vision/asl-letters/``::

        (x_train, y_train), (x_test, y_test) = load_data(
            "data/computer-vision/asl-letters", folders=["A", "B", "C"]
        )
        # x_train.shape -> (n, 50, 50, 3), dtype float32, values in [0, 1]

    This example is shown as a literal block rather than a doctest because it
    needs the image set, which is fetched separately.
"""

from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image

__all__ = ["load_data", "load_image"]

DEFAULT_SIZE = 50


def load_image(path: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Load a single image as an RGB float32 array.

    Args:
        path: Path to the image file.
        size: Side length to resize to. The image is squashed, not cropped, so
            aspect ratio is not preserved - acceptable for a fixed-input CNN.

    Returns:
        numpy.ndarray: Array of shape ``(size, size, 3)``, dtype ``float32``,
        scaled to ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with Image.open(path) as handle:
        resized = handle.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0


def load_data(
    container_path: str,
    folders: list[str] | None = None,
    size: int = DEFAULT_SIZE,
    test_split: float = 0.2,
    seed: int = 0,
    limit: int | None = 2000,
) -> tuple:
    """Load a letter-classification dataset from a folder of class subfolders.

    Args:
        container_path: Directory holding one subfolder per class.
        folders: Class names in label order. Defaults to ``["A", "B", "C"]``.
        size: Side length to resize images to.
        test_split: Fraction reserved for the test set, in ``[0, 1)``.
        seed: Seed for shuffling, so the split is reproducible.
        limit: Maximum total images to load. ``None`` loads everything.

    Returns:
        tuple: ``((x_train, y_train), (x_test, y_test))`` where ``x`` has shape
        ``(n, size, size, 3)`` float32 in ``[0, 1]`` and ``y`` holds integer
        label indices matching the position of each class in ``folders``.

    Raises:
        FileNotFoundError: If ``container_path`` or a class subfolder is
            missing.
        ValueError: If ``test_split`` is outside ``[0, 1)``.
    """
    folders = folders or ["A", "B", "C"]
    if not 0.0 <= test_split < 1.0:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}")

    paths: list[str] = []
    labels: list[int] = []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(container_path, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"{folder_path} not found. Expected one subfolder per class "
                f"({', '.join(folders)}) inside {container_path}."
            )
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder_path, name))
                labels.append(label)

    if not paths:
        raise FileNotFoundError(
            f"No images found in {container_path}. Fetch the ASL set with "
            "`python ../../scripts/fetch_data.py --only asl-letters`."
        )

    paired = list(zip(paths, labels))
    random.Random(seed).shuffle(paired)
    if limit is not None:
        paired = paired[:limit]

    images = np.stack([load_image(path, size) for path, _ in paired])
    targets = np.array([label for _, label in paired], dtype=np.int32)

    cut = int(len(images) * (1.0 - test_split))
    return (
        (images[:cut], targets[:cut]),
        (images[cut:], targets[cut:]),
    )
