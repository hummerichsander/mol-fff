from collections import namedtuple

from .cosine_distance import CosineDistance
from .cross_entropy import CrossEntropy
from .mmd import MMD
from .precision import GeometricPrecision, Precision

MetricEntry = namedtuple("MetricEntry", ["metric_name", "metric_fn", "metric_weight"])


def get_metric(metric_spec: dict, accelerator: str) -> MetricEntry:
    if metric_spec["metric_type"] == "deep_graph_kernel":
        metric_spec["metric_hparams"]["accelerator"] = accelerator
        return MetricEntry(
            metric_name=metric_spec["metric_name"],
            metric_fn=DeepGraphKernel(metric_spec.get("metric_hparams", {})),
            metric_weight=metric_spec.get("metric_weight", 1),
        )
    elif metric_spec["metric_type"] == "cross_entropy":
        metric_spec["metric_hparams"]["accelerator"] = accelerator
        return MetricEntry(
            metric_name=metric_spec["metric_name"],
            metric_fn=CrossEntropy(**metric_spec.get("metric_hparams", {})),
            metric_weight=metric_spec.get("metric_weight", 1),
        )
    elif metric_spec["metric_type"] == "mmd":
        return MetricEntry(
            metric_name=metric_spec["metric_name"],
            metric_fn=MMD(**metric_spec.get("metric_hparams", {})),
            metric_weight=metric_spec.get("metric_weight", 1),
        )
    elif metric_spec["metric_type"] == "precision":
        return MetricEntry(
            metric_name=metric_spec["metric_name"],
            metric_fn=Precision(**metric_spec.get("metric_hparams", {})),
            metric_weight=metric_spec.get("metric_weight", 1),
        )
    elif metric_spec["metric_type"] == "cosine_distance":
        return MetricEntry(
            metric_name=metric_spec["metric_name"],
            metric_fn=CosineDistance(**metric_spec.get("metric_hparams", {})),
            metric_weight=metric_spec.get("metric_weight", 1),
        )
    elif metric_spec["metric_type"] == "geometric_precision":
        return MetricEntry(
            metric_name=metric_spec["metric_name"],
            metric_fn=GeometricPrecision(**metric_spec.get("metric_hparams", {})),
            metric_weight=metric_spec.get("metric_weight", 1),
        )
    else:
        raise NotImplementedError(f"Unknown metric {metric_spec['metric_type']}")
