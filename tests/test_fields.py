from io import StringIO

import numpy as np
import pandas as pd
import pytest

from chromo.components import Polymer
from chromo.fields import UniformDensityField
import chromo.marks


def test_uniform_density_field_roundtrip():
    marks = [chromo.marks.get_by_name('HP1')]
    marks = chromo.marks.make_mark_collection(marks)

    p = Polymer.straight_line_in_x('Chr-1', 10, 1, states=np.zeros((10,)),
                                   mark_names=['HP1'])

    udf = UniformDensityField([p], marks, 10, 20, 30, 40, 50, 60)
    udf_round_trip = UniformDensityField.from_file(
        StringIO(udf.to_file(None)), [p], marks
    )

    assert udf == udf_round_trip

    # ensure the code errors if the wrong number of polymers or marks is passed
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [], marks
        )
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p, p], marks
        )
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p], pd.concat([marks, marks])
        )

    # should also error if the poly/mark has the wrong name
    with pytest.raises(ValueError):
        p2 = Polymer.straight_line_in_x('Chr-1', 10, 1, states=np.zeros((10,)),
                                        mark_names=['HP1'])
        p2.name = "different"
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p2], marks
        )
    with pytest.raises(ValueError):
        marks2 = chromo.marks.make_mark_collection(marks)
        marks2.loc[0, 'name'] = 'different'
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p], marks2
        )
