# 3D Nuclei Centroid Detection Pipeline

A comprehensive pipeline for locating centroid for 3D BlastoSpim data and training ML models for automatic nuclei centroid detection.

## 🎯 Project Overview

This project implements a **two-stage approach** for 3D nuclei centroid detection:

1. **Label Generation Stage** : Extract accurate centroid labels from manually annotated 3D volumes
2. **Model Training Stage** : Train ML models to automatically predict centroids from raw volumes

## 📁 Current Project Structure

```
centroid_model_blastospim/
├── src/
│   ├── preprocessing/
│   │   ├── centroid_extractor.py      # Core 3D centroid extraction class
│   │   ├── centroid_batch_processor.py # Batch processing pipeline
│   │   └── __init__.py
│   ├── utils/
│   │   ├── config.py                  # YAML configuration management
│   │   └── __init__.py
│   └── __init__.py
├── configs/
│   └── base_config.yaml              # Main configuration (voxel spacing, etc.)
├── notebooks/
│   ├── single_file_processing.ipynb   # Interactive single file processing
│   └── folder_processing.ipynb        # Interactive batch processing
├── data/
│   ├── raw/                          # Raw BlastoSpim NPZ files
│   │   └── folders/                  # Example raw data file
│   │       └── foldername_[num].npz/            # File with img and labels arrays
│   └── labels/                       # Generated centroid labels
│       └── label_[filename]/         # Organized output per file
│           ├── data/
│           ├── visualizations/
│           ├── analysis/
│           ├── metadata/
│           └── logs/
│── models/                           # Checkpointed models and model weights
├── docs/                             # Detailed documentation
├── tests/                            # Unit tests
├── requirements.txt
└── README.md
```


## 🚀 Quick Start (Label Generation)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Single File Processing
```python
from src.preprocessing.centroid_batch_processor import SingleFileProcessor

# Initialize processor with configuration
processor = SingleFileProcessor(output_dir="data/labels")

# Process a single NPZ file
result = processor.process(
    input_file="path/to/your/blastospim.npz",
    create_visualization=True
)

print(f"Centroid: {result['centroid_data']['centroid']}")
```

### Batch Processing
```python
from src.preprocessing.centroid_batch_processor import CentroidBatchProcessor

# Initialize batch processor
batch_processor = CentroidBatchProcessor(config_path="configs/base_config.yaml")

# Process entire directory
results = batch_processor.process_directory(
    input_dir="data/raw_npz",
    file_pattern="*.npz"
)

print(f"Processed {len(results['results'])} files")
```

### Using Jupyter Notebooks
- **Single File**: `/notebooks/single_file_processing.ipynb`
- **Batch Processing**: `/notebooks/folder_processing.ipynb`

## 📊 Current Pipeline Output

Each processed file generates:
```
data/labels/label_[filename]/
├── data/label.npz                      # Processed volume + centroid data
├── visualizations/comprehensive_analysis.png  # Multi-panel visualization
├── analysis/detailed_analysis.json     # Detailed metrics
├── metadata/metadata.json             # Processing metadata
└── logs/                              # Processing logs
```



**Current Status**: ✅ **Label Generation Pipeline Complete**  
**Next Milestone**: 🔄 **ML Model Training Implementation**
