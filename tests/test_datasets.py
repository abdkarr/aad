from pathlib import Path

import numpy as np

from aad import datasets

PROJECT_DIR = str(Path(__file__).parents[1])

def test_read_rte(tmp_path):
    response_mat, gt_labels = datasets.read_rte(tmp_path)

    assert response_mat.shape == (164, 800), "Wrong shape for response matrix."
    assert len(gt_labels) == 800, "Wrong shape for ground truth labels array."
    
    class_ids = np.unique(gt_labels)
    assert len(class_ids) == 2, "Wrong number of classes."
    assert len(np.setdiff1d(class_ids, [1, 2])) == 0, "Wrong class ids."