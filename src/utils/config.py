"""
Configuration utilities for loading and managing YAML configurations
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict or {}


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation (e.g., 'data.raw_data.source')"""
    keys = key.split(".")
    value = config

    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set nested configuration value using dot notation"""
    keys = key.split(".")
    target = config

    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]

    # Set the final value
    target[keys[-1]] = value


def merge_configs(
    base_config: Dict[str, Any], updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries"""
    merged = base_config.copy()

    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys exist in configuration"""
    missing_keys = []

    for key in required_keys:
        if get_config_value(config, key) is None:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


class ConfigManager:
    """
    Configuration manager class for handling YAML configurations
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to main configuration file
        """
        self.config = {}
        self.config_path = None

        if config_path:
            self.load(config_path)

    def load(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file"""
        self.config_path = Path(config_path)
        self.config = load_config(config_path)

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file"""
        if output_path is None:
            if self.config_path is None:
                raise ValueError(
                    "No output path provided and no original config path available"
                )
            output_path = self.config_path

        save_config(self.config, output_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return get_config_value(self.config, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        set_config_value(self.config, key, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary"""
        self.config = merge_configs(self.config, updates)

    def validate(self, required_keys: List[str]) -> None:
        """Validate configuration has required keys"""
        validate_config(self.config, required_keys)

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()

    def get_data_paths(self) -> Dict[str, str]:
        """Get all data-related paths from configuration"""
        return {
            "base_dir": self.get("data.base_dir", "./data"),
            "raw_source": self.get("data.raw_data.source", "./data/raw"),
            "labels_source": self.get("data.labels.source", "./data/labels"),
            "output_base": self.get("data.output.base_dir", "./output"),
            "results": self.get("data.output.results", "./output/results"),
            "visualizations": self.get(
                "data.output.visualizations", "./output/visualizations"
            ),
            "logs": self.get("data.output.logs", "./output/logs"),
        }

    def get_centroid_config(self) -> Dict[str, Any]:
        """Get centroid extraction configuration"""
        return {
            "method": self.get("centroid_extraction.method", "center_of_mass"),
            "connectivity": self.get("centroid_extraction.connectivity", 3),
            "coordinate_order": self.get("centroid_extraction.coordinate_order", "zyx"),
            "voxel_size": {
                "z": self.get(
                    "centroid_extraction.voxel_size.z", 2.0
                ),  # BlastoSpim Z-axis resolution
                "y": self.get(
                    "centroid_extraction.voxel_size.y", 0.208
                ),  # BlastoSpim Y-axis resolution
                "x": self.get(
                    "centroid_extraction.voxel_size.x", 0.208
                ),  # BlastoSpim X-axis resolution
            },
            "filtering_enabled": self.get(
                "centroid_extraction.filtering.enabled", False
            ),
            "min_volume": self.get("centroid_extraction.filtering.min_volume", 10),
            "max_volume": self.get("centroid_extraction.filtering.max_volume", 10000),
        }

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return {
            "enabled": self.get("visualization.enabled", True),
            "figure_width": self.get("visualization.figure.width", 20),
            "figure_height": self.get("visualization.figure.height", 12),
            "dpi": self.get("visualization.figure.dpi", 300),
            "format": self.get("visualization.figure.format", "png"),
            "volume_cmap": self.get("visualization.colormaps.volume", "gray"),
            "mask_cmap": self.get("visualization.colormaps.mask", "viridis"),
            "overlay_alpha": self.get("visualization.colormaps.overlay_alpha", 0.5),
            "max_3d_points": self.get("visualization.scatter_3d.max_points", 5000),
            "centroid_size": self.get("visualization.scatter_3d.centroid_size", 100),
            "centroid_color": self.get(
                "visualization.scatter_3d.centroid_color", "yellow"
            ),
        }
