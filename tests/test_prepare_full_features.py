from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location("prepare_full_features", Path(__file__).parents[1] / "scripts/prepare_full_features.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_MODULE)
completed_chunks, validate_resume = _MODULE.completed_chunks, _MODULE.validate_resume


def test_resume_rejects_changed_completed_feature_chunk():
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    progress = {"schema_version": 2, "completed_images": 2, "total_images": 3,
                "array_shape": [3, 8], "array_dtype": "float32", "chunks": completed_chunks(values, 2, 1)}
    values[0, 0] = 999
    with pytest.raises(ValueError, match="chunk hash"):
        validate_resume(progress, values, total_images=3)


@pytest.mark.parametrize("completed", [-1, 4])
def test_resume_rejects_out_of_bounds_progress(completed):
    values = np.zeros((3, 8), dtype=np.float32)
    progress = {"schema_version": 2, "completed_images": completed, "total_images": 3,
                "array_shape": [3, 8], "array_dtype": "float32", "chunks": []}
    with pytest.raises(ValueError, match="completed_images"):
        validate_resume(progress, values, total_images=3)
