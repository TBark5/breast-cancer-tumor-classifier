import numpy as np
from src.data import split_data

def test_dimensions_mapping_and_quality(dataset):
    frame, features, _, _ = dataset
    assert frame[features].shape == (569, 30)
    assert frame.target.value_counts().to_dict() == {0: 357, 1: 212}
    assert (frame.target == 1 - frame.original_target).all()
    assert (frame.loc[frame.original_target == 0, "label"] == "Malignant").all()
    assert not frame[features].isna().any().any()
    assert not frame[features].duplicated().any()
    assert (frame[features].nunique() > 1).all()

def test_split_is_reproducible_disjoint_stratified(dataset):
    frame, _, train, test = dataset
    _, _, train2, test2 = split_data()
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(test, test2)
    assert (len(train), len(test)) == (455, 114)
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(frame.index)
    assert frame.loc[test, "target"].value_counts().to_dict() == {0: 72, 1: 42}
