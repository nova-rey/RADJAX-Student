import pytest

from radjax_student.learning import MetricRecord
from radjax_student.learning.telemetry import MetricRetentionPolicy, MetricSeries


def test_metric_series_is_bounded_and_summary_keeps_all_observations():
    series = MetricSeries("loss", MetricRetentionPolicy(max_records=2))
    for step, value in enumerate((3.0, 2.0, 1.0)):
        series.add(MetricRecord("loss", value, step))
    assert [r.value for r in series.records] == [2.0, 1.0]
    assert series.summary().mean == 2.0 and series.summary().count == 3
    with pytest.raises(ValueError):
        series.add(MetricRecord("other", 1, 0))
