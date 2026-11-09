"""Pure-Force EMG prediction experiments.

The configurable loader is re-exported lazily so both
``from emg2force import load_data`` and the more explicit
``from emg2force.data_api import load_data`` are supported.
"""

__version__ = "0.1.0"

_DATA_API_EXPORTS = {
    "FilterConfig",
    "SPLITS",
    "EMGFilter",
    "apply_emg_filter",
    "NormalizationStats",
    "NormStats",
    "load_norm_stats",
    "DatasetSpec",
    "DatasetRegistry",
    "DEFAULT_DATASETS",
    "register_dataset",
    "list_datasets",
    "CachePaths",
    "processed_cache_paths",
    "cache_paths_for",
    "ProcessedCache",
    "ProcessedWindowDataset",
    "EmptyDataset",
    "DistributedEvalBatchSampler",
    "DataBundle",
    "load_data",
    "build_dataloaders",
    "get_data_loaders",
    "build_data",
    "process_data",
    "load_dataset",
    "DataLoaderAPI",
    "DataAPI",
    "DatasetLoader",
    "DataLoaders",
    "DataModule",
}


def __getattr__(name: str):
    if name in _DATA_API_EXPORTS:
        from . import data_api

        return getattr(data_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", *_DATA_API_EXPORTS]
