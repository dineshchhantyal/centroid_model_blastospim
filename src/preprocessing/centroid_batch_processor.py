"""
Batch Centroid Processor

This module provides an OOP-based batch processor for extracting centroids
from 3D volume data and organizing the results in a structured format.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import logging
from datetime import datetime
import json
from tqdm import tqdm
from src.utils.config import ConfigManager

import sys

project_root = Path("/mnt/home/dchhantyal/centroid_model_blastospim")
from src.preprocessing.centroid_extractor import CentroidExtractor


class CentroidBatchProcessor:
    """
    Batch processor for extracting centroids from multiple 3D volume files.

    This class provides comprehensive functionality for:
    1. Processing single files or entire directories
    2. Organizing output in structured directories
    3. Creating comprehensive visualizations
    4. Generating summary reports
    5. Error handling and logging
    """

    def __init__(
        self,
        output_base_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            output_base_dir: Base directory for saving processed results (defaults from config)
            config_path: Path to configuration file (defaults to configs/base_config.yaml)
            logger: Optional logger instance
        """
        # Load configuration first
        if config_path is None:
            # Resolve config path relative to this script's directory
            config_path = project_root / "configs" / "base_config.yaml"

        self.config_manager = ConfigManager(config_path)

        # Set output directory from config if not provided
        if output_base_dir is None:
            output_base_dir = self.config_manager.get(
                "data.labels.base_dir", "data/labels"
            )

        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        if logger is None:
            log_level = self.config_manager.get("logging.level", "INFO")
            log_format = self.config_manager.get(
                "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            logging.basicConfig(level=getattr(logging, log_level), format=log_format)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Initialize centroid extractor
        self.extractor = CentroidExtractor(logger=self.logger)

        # Get BlastoSpim dataset voxel spacing from config (research paper values)
        voxel_config = self.config_manager.get("centroid_extraction.voxel_size")
        self.blastospim_voxel_spacing = (
            voxel_config["z"],  # Z-axis (between slices): 2.0 µm
            voxel_config["y"],  # Y-axis (in-plane): 0.208 µm
            voxel_config["x"],  # X-axis (in-plane): 0.208 µm
        )

        self.logger.info(
            f"Using voxel spacing from config: Z={self.blastospim_voxel_spacing[0]}µm, "
            f"Y={self.blastospim_voxel_spacing[1]}µm, X={self.blastospim_voxel_spacing[2]}µm"
        )

        # Track processing statistics
        self.processing_stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "start_time": None,
            "end_time": None,
            "files_processed": [],
            "errors": [],
        }

    def create_output_structure(self, filename: str) -> Dict[str, Path]:
        """
        Create organized output directory structure for a file.

        Args:
            filename: Name of the input file (without extension)

        Returns:
            Dictionary containing all output paths
        """
        # Create main directory for this file using config pattern

        folder_pattern = self.config_manager.get(
            "data.labels.folder_pattern", "label_{raw_data_filename}"
        )
        file_dir = self.output_base_dir / folder_pattern.format(
            raw_data_filename=filename
        )
        file_dir.mkdir(parents=True, exist_ok=True)

        # Get subdirectory names from config
        viz_dir_name = self.config_manager.get(
            "data.labels.visualizations_dir", "visualizations"
        )
        logs_dir_name = self.config_manager.get("data.labels.logs_dir", "logs")

        # Create subdirectories
        paths = {
            "main_dir": file_dir,
            "data_dir": file_dir / "data",
            "visualization_dir": file_dir / viz_dir_name,
            "analysis_dir": file_dir / "analysis",
            "metadata_dir": file_dir / "metadata",
            "logs_dir": file_dir / logs_dir_name,
        }

        # Create all subdirectories
        for path in paths.values():
            if path != paths["main_dir"]:
                path.mkdir(parents=True, exist_ok=True)

        return paths

    def process_single_file(
        self,
        input_file: Union[str, Path],
        output_paths: Optional[Dict[str, Path]] = None,
        create_visualization: Optional[bool] = None,
        save_metadata: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process a single NPZ file and save results in organized structure.

        Args:
            input_file: Path to input NPZ file
            create_visualization: Whether to create comprehensive visualization (defaults from config)
            save_metadata: Whether to save processing metadata (defaults from config)

        Returns:
            Dictionary containing processing results
        """
        # Use config defaults if not specified
        if create_visualization is None:
            create_visualization = self.config_manager.get(
                "visualization.enabled", True
            )
        if save_metadata is None:
            save_metadata = self.config_manager.get(
                "analysis.export.include_metadata", True
            )

        input_path = Path(input_file)
        filename = input_path.stem

        self.logger.info(f"Processing file: {filename}")

        try:
            # Create output structure
            output_paths = (
                self.create_output_structure(filename)
                if output_paths is None
                else output_paths
            )

            # Load the NPZ file
            self.logger.info(f"Loading data from: {input_path}")
            data = np.load(input_path)

            # Validate data structure using config
            expected_keys = self.config_manager.get(
                "data.raw_data.expected_keys", ["img", "labels"]
            )
            for key in expected_keys:
                if key not in data:
                    raise ValueError(f"NPZ file must contain '{key}' key")

            volume = data["img"]
            mask = data["labels"]

            self.logger.info(f"Volume shape: {volume.shape}")
            self.logger.info(f"Mask shape: {mask.shape}")
            self.logger.info(f"Number of objects: {len(np.unique(mask)) - 1}")

            # Extract centroid using our extractor
            centroid_data = self.extractor.extract_center_of_mass(mask)

            # Save processed label data using config filename
            label_filename = self.config_manager.get(
                "data.labels.label_file", "label.npz"
            )
            label_output_path = output_paths["data_dir"] / label_filename
            np.savez_compressed(
                label_output_path,
                centroid=centroid_data["centroid"],
                bounding_box=centroid_data["bounding_box"],
            )

            self.logger.info(f"Saved label data to: {label_output_path}")

            # Create comprehensive visualization
            if create_visualization:
                viz_path = (
                    output_paths["visualization_dir"] / "comprehensive_analysis.png"
                )
                # Use correct BlastoSpim voxel spacing from research paper
                self.extractor.create_comprehensive_visualization(
                    volume=volume,
                    mask=mask,
                    centroid_data=centroid_data,
                    save_path=str(viz_path),
                    voxel_size=self.blastospim_voxel_spacing,  # (z=2.0µm, y/x=0.208µm)
                )

            # Save detailed analysis
            analysis_data = self._create_detailed_analysis(
                volume, mask, centroid_data, filename
            )

            analysis_path = output_paths["analysis_dir"] / "detailed_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(analysis_data, f, indent=2, default=str)

            # Save metadata
            if save_metadata:
                metadata = self._create_metadata(
                    input_path, output_paths, centroid_data, analysis_data
                )

                metadata_filename = self.config_manager.get(
                    "data.labels.metadata_file", "metadata.json"
                )
                metadata_path = output_paths["metadata_dir"] / metadata_filename
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            # Update processing statistics
            self.processing_stats["successful_files"] += 1
            self.processing_stats["files_processed"].append(
                {
                    "filename": filename,
                    "status": "success",
                    "centroid": centroid_data["centroid"].tolist(),
                    "num_objects": centroid_data["num_objects"],
                    "output_dir": str(output_paths["main_dir"]),
                }
            )

            self.logger.info(f"Successfully processed {filename}")

            return {
                "status": "success",
                "filename": filename,
                "centroid_data": centroid_data,
                "output_paths": output_paths,
                "analysis_data": analysis_data,
            }

        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            self.logger.error(error_msg)

            # Update error statistics
            self.processing_stats["failed_files"] += 1
            self.processing_stats["errors"].append(
                {
                    "filename": filename,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return {"status": "error", "filename": filename, "error": str(e)}

    def process_directory(
        self,
        input_dir: Union[str, Path],
        file_pattern: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process all NPZ files in a directory.

        Args:
            input_dir: Directory containing NPZ files
            file_pattern: File pattern to match (defaults from config)
            max_files: Maximum number of files to process (None for all)

        Returns:
            Dictionary containing batch processing results
        """
        # Use config default for file pattern if not specified
        if file_pattern is None:
            file_pattern = self.config_manager.get(
                "data.raw_data.file_pattern", "*.npz"
            )

        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")

        # Find all matching files
        files = list(input_path.glob(file_pattern))

        if max_files:
            files = files[:max_files]

        self.logger.info(
            f"Found {len(files)} files to process in {input_path} using pattern '{file_pattern}'"
        )

        # Initialize processing statistics
        self.processing_stats["total_files"] = len(files)
        self.processing_stats["start_time"] = datetime.now()

        # Process each file with progress bar
        results = []
        for file_path in tqdm(files, desc="Processing files"):
            result = self.process_single_file(file_path)
            results.append(result)

        self.processing_stats["end_time"] = datetime.now()

        # Create batch summary
        summary = self._create_batch_summary()

        # Save batch summary
        summary_path = self.output_base_dir / "batch_processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Batch processing complete. Summary saved to: {summary_path}")

        return {
            "summary": summary,
            "results": results,
            "output_dir": self.output_base_dir,
        }

    def _create_detailed_analysis(self, volume, mask, centroid_data, filename):
        """Create detailed analysis data."""
        return {
            "filename": filename,
            "processing_timestamp": datetime.now().isoformat(),
            "volume_info": {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "min_intensity": float(np.min(volume)),
                "max_intensity": float(np.max(volume)),
                "mean_intensity": float(np.mean(volume)),
            },
            "mask_info": {
                "shape": mask.shape,
                "dtype": str(mask.dtype),
                "unique_labels": len(np.unique(mask)),
                "background_voxels": int(np.sum(mask == 0)),
                "foreground_voxels": int(np.sum(mask > 0)),
            },
            "centroid_analysis": {
                "centroid_coordinates": centroid_data["centroid"].tolist(),
                "bounding_box": centroid_data["bounding_box"],
                "total_volume": centroid_data["total_volume"],
                "num_objects": centroid_data["num_objects"],
                "volume_density": centroid_data["total_volume"] / np.prod(volume.shape),
            },
        }

    def _create_metadata(self, input_path, output_paths, centroid_data, analysis_data):
        """Create processing metadata."""
        return {
            "processing_info": {
                "processor_version": "1.0.0",
                "processing_timestamp": datetime.now().isoformat(),
                "input_file": str(input_path),
                "output_directory": str(output_paths["main_dir"]),
            },
            "file_structure": {
                "label_file": str(output_paths["data_dir"] / "label.npz"),
                "visualization_files": [
                    str(
                        output_paths["visualization_dir"] / "comprehensive_analysis.png"
                    ),
                    str(output_paths["visualization_dir"] / "centroid_overlay.png"),
                ],
                "analysis_file": str(
                    output_paths["analysis_dir"] / "detailed_analysis.json"
                ),
            },
            "processing_results": {
                "centroid": centroid_data["centroid"].tolist(),
                "num_objects": centroid_data["num_objects"],
                "total_volume": centroid_data["total_volume"],
            },
        }

    def _create_batch_summary(self):
        """Create batch processing summary."""
        duration = (
            self.processing_stats["end_time"] - self.processing_stats["start_time"]
        )

        return {
            "batch_info": {
                "total_files": self.processing_stats["total_files"],
                "successful_files": self.processing_stats["successful_files"],
                "failed_files": self.processing_stats["failed_files"],
                "success_rate": (
                    self.processing_stats["successful_files"]
                    / self.processing_stats["total_files"]
                    if self.processing_stats["total_files"] > 0
                    else 0
                ),
                "processing_duration": str(duration),
                "start_time": self.processing_stats["start_time"].isoformat(),
                "end_time": self.processing_stats["end_time"].isoformat(),
            },
            "output_directory": str(self.output_base_dir),
            "files_processed": self.processing_stats["files_processed"],
            "errors": self.processing_stats["errors"],
        }


class SingleFileProcessor:
    """
    Simplified processor for handling a single file with the CentroidExtractor.

    This class provides a clean interface for processing individual files
    with comprehensive output organization.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/labels",
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the single file processor."""
        self.batch_processor = CentroidBatchProcessor(
            output_base_dir=output_dir, config_path=config_path
        )
        self.logger = self.batch_processor.logger

    def process(
        self, input_file: Union[str, Path], create_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single file and return results.

        Args:
            input_file: Path to input NPZ file
            create_visualization: Whether to create visualizations

        Returns:
            Processing results dictionary
        """
        return self.batch_processor.process_single_file(
            input_file=input_file, create_visualization=create_visualization
        )


# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Process single file
    print("=== Single File Processing Example ===")

    # Initialize processor
    processor = SingleFileProcessor(output_dir=project_root / "test-data/labels")

    # Process the specific file
    input_file = "/mnt/home/dchhantyal/ceph/datasets/Blast_001.npz"

    if Path(input_file).exists():
        result = processor.process(input_file, create_visualization=True)

        if result["status"] == "success":
            print(f"✅ Successfully processed: {result['filename']}")
            print(f"📍 Centroid: {result['centroid_data']['centroid']}")
            print(f"🔢 Number of objects: {result['centroid_data']['num_objects']}")
            print(f"📁 Output directory: {result['output_paths']['main_dir']}")
        else:
            print(f"❌ Failed to process: {result['error']}")
    else:
        print(f"❌ File not found: {input_file}")
