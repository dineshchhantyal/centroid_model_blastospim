"""
3D Centroid Extractor with Visualization

This module provides a class for extracting the center of mass
from 3D volume labels with visualization capabilities.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CentroidExtractor:
    """
    Class for extracting center of mass from 3D volume labels.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the CentroidExtractor."""
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def extract_center_of_mass(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract center of mass (geometric center) from 3D mask.

        Args:
            mask: 3D labeled mask array

        Returns:
            Dictionary containing:
                - centroid: (z, y, x) center of mass coordinates
                - bounding_box: min/max coordinates of all objects
                - total_volume: number of foreground voxels
                - num_objects: number of labeled objects
        """
        # Get all non-background voxels
        foreground_mask = mask > 0

        if not np.any(foreground_mask):
            self.logger.warning("No foreground objects found in mask")
            return {
                "centroid": np.array([0.0, 0.0, 0.0]),
                "bounding_box": None,
                "total_volume": 0,
                "num_objects": 0,
            }

        # Calculate center of mass (simple geometric center)
        coords = np.where(foreground_mask)
        centroid = np.array(
            [np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])]  # Z  # Y  # X
        )

        # Calculate bounding box, storing for future reference
        bbox = {
            "z_min": int(np.min(coords[0])),
            "z_max": int(np.max(coords[0])),
            "y_min": int(np.min(coords[1])),
            "y_max": int(np.max(coords[1])),
            "x_min": int(np.min(coords[2])),
            "x_max": int(np.max(coords[2])),
        }

        # Count objects
        unique_labels = np.unique(mask[foreground_mask])
        num_objects = len(unique_labels)

        return {
            "centroid": centroid.astype(np.float32),
            "bounding_box": bbox,
            "total_volume": int(np.sum(foreground_mask)),
            "num_objects": num_objects,
        }

    def create_comprehensive_visualization(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        centroid_data: Dict[str, Any],
        save_path: Optional[str] = None,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Create comprehensive visualization including bounding box and centroid.

        Args:
            volume: 3D intensity volume
            mask: 3D labeled mask
            centroid_data: Output from extract_center_of_mass()
            save_path: Path to save the visualization
            voxel_size: Physical voxel size (z, y, x) in micrometers
        """
        centroid = centroid_data["centroid"]
        bbox = centroid_data["bounding_box"]

        if bbox is None:
            self.logger.error("No bounding box data available")
            return

        # Calculate physical coordinates
        physical_centroid = centroid * np.array(voxel_size)

        # Create comprehensive figure with more space for text
        fig = plt.figure(figsize=(24, 14))

        # Main title
        fig.suptitle(
            "Comprehensive Geometric Center Analysis - Bounding Box Method",
            fontsize=16,
            fontweight="bold",
        )

        # Get middle slices
        z_mid = int(centroid[0])
        y_mid = int(centroid[1])
        x_mid = int(centroid[2])

        # 1. Z slice (middle) with bounding box
        ax1 = plt.subplot(3, 4, 1)
        z_slice = volume[z_mid, :, :]
        z_mask = mask[z_mid, :, :]
        ax1.imshow(z_slice, cmap="gray", alpha=0.7)
        ax1.imshow(z_mask, cmap="viridis", alpha=0.5)
        ax1.plot(centroid[2], centroid[1], "w*", markersize=15, markeredgecolor="black")

        # Add bounding box
        rect = patches.Rectangle(
            (bbox["x_min"], bbox["y_min"]),
            bbox["x_max"] - bbox["x_min"],
            bbox["y_max"] - bbox["y_min"],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
        ax1.add_patch(rect)
        ax1.set_title(f"Z slice {z_mid} (Middle)\nwith Bounding Box", fontsize=10)
        ax1.text(50, 100, "Centroid", color="white", fontweight="bold")

        # 2. Y slice (middle)
        ax2 = plt.subplot(3, 4, 2)
        y_slice = volume[:, y_mid, :]
        y_mask = mask[:, y_mid, :]
        ax2.imshow(y_slice, cmap="gray", alpha=0.7)
        ax2.imshow(y_mask, cmap="viridis", alpha=0.5)
        ax2.plot(centroid[2], centroid[0], "w*", markersize=15, markeredgecolor="black")
        rect = patches.Rectangle(
            (bbox["x_min"], bbox["z_min"]),
            bbox["x_max"] - bbox["x_min"],
            bbox["z_max"] - bbox["z_min"],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
        ax2.add_patch(rect)
        ax2.set_title(f"Y slice {y_mid} (Middle)\nwith Bounding Box", fontsize=10)

        # 3. X slice (middle)
        ax3 = plt.subplot(3, 4, 3)
        x_slice = volume[:, :, x_mid]
        x_mask = mask[:, :, x_mid]
        ax3.imshow(x_slice, cmap="gray", alpha=0.7)
        ax3.imshow(x_mask, cmap="viridis", alpha=0.5)
        ax3.plot(centroid[1], centroid[0], "w*", markersize=15, markeredgecolor="black")
        rect = patches.Rectangle(
            (bbox["y_min"], bbox["z_min"]),
            bbox["y_max"] - bbox["y_min"],
            bbox["z_max"] - bbox["z_min"],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
        ax3.add_patch(rect)
        ax3.set_title(f"X slice {x_mid} (Middle)\nwith Bounding Box", fontsize=10)

        # 4. 3D Volume & Bounding Box
        ax4 = plt.subplot(3, 4, 4, projection="3d")

        # Sample points from mask for 3D visualization
        foreground_coords = np.where(mask > 0)
        if len(foreground_coords[0]) > 5000:  # Subsample for performance
            idx = np.random.choice(len(foreground_coords[0]), 5000, replace=False)
            z_points = foreground_coords[0][idx]
            y_points = foreground_coords[1][idx]
            x_points = foreground_coords[2][idx]
        else:
            z_points, y_points, x_points = foreground_coords

        ax4.scatter(x_points, y_points, z_points, c="red", alpha=0.1, s=1)
        ax4.scatter(
            centroid[2], centroid[1], centroid[0], c="yellow", s=100, marker="*"
        )

        # Draw bounding box edges
        edges = [
            # Bottom face
            (
                [bbox["x_min"], bbox["x_max"]],
                [bbox["y_min"], bbox["y_min"]],
                [bbox["z_min"], bbox["z_min"]],
            ),
            (
                [bbox["x_min"], bbox["x_min"]],
                [bbox["y_min"], bbox["y_max"]],
                [bbox["z_min"], bbox["z_min"]],
            ),
            (
                [bbox["x_max"], bbox["x_min"]],
                [bbox["y_max"], bbox["y_max"]],
                [bbox["z_min"], bbox["z_min"]],
            ),
            (
                [bbox["x_max"], bbox["x_max"]],
                [bbox["y_max"], bbox["y_min"]],
                [bbox["z_min"], bbox["z_min"]],
            ),
            # Top face
            (
                [bbox["x_min"], bbox["x_max"]],
                [bbox["y_min"], bbox["y_min"]],
                [bbox["z_max"], bbox["z_max"]],
            ),
            (
                [bbox["x_min"], bbox["x_min"]],
                [bbox["y_min"], bbox["y_max"]],
                [bbox["z_max"], bbox["z_max"]],
            ),
            (
                [bbox["x_max"], bbox["x_min"]],
                [bbox["y_max"], bbox["y_max"]],
                [bbox["z_max"], bbox["z_max"]],
            ),
            (
                [bbox["x_max"], bbox["x_max"]],
                [bbox["y_max"], bbox["y_min"]],
                [bbox["z_max"], bbox["z_max"]],
            ),
            # Vertical edges
            (
                [bbox["x_min"], bbox["x_min"]],
                [bbox["y_min"], bbox["y_min"]],
                [bbox["z_min"], bbox["z_max"]],
            ),
            (
                [bbox["x_max"], bbox["x_max"]],
                [bbox["y_min"], bbox["y_min"]],
                [bbox["z_min"], bbox["z_max"]],
            ),
            (
                [bbox["x_min"], bbox["x_min"]],
                [bbox["y_max"], bbox["y_max"]],
                [bbox["z_min"], bbox["z_max"]],
            ),
            (
                [bbox["x_max"], bbox["x_max"]],
                [bbox["y_max"], bbox["y_max"]],
                [bbox["z_min"], bbox["z_max"]],
            ),
        ]

        for i, (x, y, z) in enumerate(edges):
            ax4.plot(x, y, z, "g-", linewidth=2, alpha=0.8)

        ax4.set_title("3D Volume & Bounding Box", fontsize=10)
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")

        # 5. Centroid Z slice
        ax5 = plt.subplot(3, 4, 5)
        ax5.imshow(volume[z_mid, :, :], cmap="gray")
        contours = measure.find_contours(mask[z_mid, :, :], 0.5)
        for contour in contours:
            ax5.plot(contour[:, 1], contour[:, 0], "cyan", linewidth=1)
        ax5.plot(centroid[2], centroid[1], "w*", markersize=15)
        ax5.set_title(f"Centroid Z slice {z_mid}", fontsize=10)
        ax5.text(50, 150, "Centroid lines", color="cyan", fontweight="bold")

        # 6-7. Y and X centroid slices
        ax6 = plt.subplot(3, 4, 6)
        ax6.imshow(volume[:, y_mid, :], cmap="gray")
        ax6.plot(centroid[2], centroid[0], "w*", markersize=15)
        ax6.set_title(f"Centroid Y slice {y_mid}", fontsize=10)

        ax7 = plt.subplot(3, 4, 7)
        ax7.imshow(volume[:, :, x_mid], cmap="gray")
        ax7.plot(centroid[1], centroid[0], "w*", markersize=15)
        ax7.set_title(f"Centroid X slice {x_mid}", fontsize=10)

        # 8. Nuclei Overlay (Z projection)
        ax8 = plt.subplot(3, 4, 8)
        z_projection = np.max(volume, axis=0)
        mask_projection = np.max(mask, axis=0)
        ax8.imshow(z_projection, cmap="gray", alpha=0.7)
        ax8.imshow(mask_projection, cmap="viridis", alpha=0.5)
        ax8.plot(centroid[2], centroid[1], "w*", markersize=15)
        ax8.set_title(f"Nuclei Overlay Z={z_mid}", fontsize=10)

        # 9-12. Min/Max slices
        ax9 = plt.subplot(3, 4, 9)
        ax9.imshow(volume[bbox["z_min"], :, :], cmap="gray")
        ax9.plot(centroid[2], centroid[1], "r*", markersize=10)
        ax9.set_title(f'Z min slice {bbox["z_min"]}', fontsize=10)

        ax10 = plt.subplot(3, 4, 10)
        ax10.imshow(volume[bbox["z_max"], :, :], cmap="gray")
        ax10.plot(centroid[2], centroid[1], "r*", markersize=10)
        ax10.set_title(f'Z max slice {bbox["z_max"]}', fontsize=10)

        ax11 = plt.subplot(3, 4, 11)
        ax11.imshow(volume[:, bbox["y_min"], :], cmap="gray")
        ax11.plot(centroid[2], centroid[0], "r*", markersize=10)
        ax11.set_title(f'Y min slice {bbox["y_min"]}', fontsize=10)

        ax12 = plt.subplot(3, 4, 12)
        ax12.imshow(volume[:, bbox["y_max"], :], cmap="gray")
        ax12.plot(centroid[2], centroid[0], "r*", markersize=10)
        ax12.set_title(f'Y max slice {bbox["y_max"]}', fontsize=10)

        # Add comprehensive statistics text
        stats_text = f"""GEOMETRIC CENTER ANALYSIS:
Volume shape: {volume.shape}
Individual nuclei: {centroid_data['num_objects']}
Total nuclei volume: {centroid_data['total_volume']:,} voxels
Nuclei density: {centroid_data['total_volume']/np.prod(volume.shape)*100:.1f}%

BOUNDING BOX:
Z: ({bbox['z_min']}, {bbox['z_max']}) [size: {bbox['z_max']-bbox['z_min']+1}]
Y: ({bbox['y_min']}, {bbox['y_max']}) [size: {bbox['y_max']-bbox['y_min']+1}]
X: ({bbox['x_min']}, {bbox['x_max']}) [size: {bbox['x_max']-bbox['x_min']+1}]

CENTROID (voxels):
Z: {centroid[0]:.2f}
Y: {centroid[1]:.2f}
X: {centroid[2]:.2f}

CENTROID (physical):
Z: {physical_centroid[0]:.2f} μm
Y: {physical_centroid[1]:.2f} μm
X: {physical_centroid[2]:.2f} μm"""

        # Add text box with statistics - positioned to avoid overlap
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        fig.text(
            0.82,  # Move further right
            0.75,  # Move higher up
            stats_text,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
            family="monospace",
        )

        # Add method explanation - repositioned to avoid overlap
        method_text = f"""BOUNDING BOX CALCULATION:

1. Find all non-zero voxels
2. Get min/max coordinates:
   Z: min={bbox['z_min']}, max={bbox['z_max']}
   Y: min={bbox['y_min']}, max={bbox['y_max']}
   X: min={bbox['x_min']}, max={bbox['x_max']}

3. Calculate center:
   Z: ({bbox['z_min']} + {bbox['z_max']})/2 = {(bbox['z_min'] + bbox['z_max'])/2:.1f}
   Y: ({bbox['y_min']} + {bbox['y_max']})/2 = {(bbox['y_min'] + bbox['y_max'])/2:.1f}
   X: ({bbox['x_min']} + {bbox['x_max']})/2 = {(bbox['x_min'] + bbox['x_max'])/2:.1f}

Volume Analysis Method"""

        fig.text(
            0.82,  # Move further right
            0.35,  # Lower position
            method_text,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            family="monospace",
        )

        # Adjust layout to accommodate text boxes
        plt.subplots_adjust(
            left=0.05, right=0.80, top=0.92, bottom=0.08, hspace=0.3, wspace=0.3
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Comprehensive visualization saved to: {save_path}")

        plt.show()

    def process_npz_file(
        self, file_path: str, save_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Process an NPZ file and extract center of mass with comprehensive visualization.

        Args:
            file_path: Path to NPZ file containing 'img' and 'labels' keys
            save_visualization: Whether to create and save visualization

        Returns:
            Dictionary with centroid data and file information
        """
        try:
            # Load data
            self.logger.info(f"Loading {file_path}...")
            data = np.load(file_path)

            if "img" not in data or "labels" not in data:
                raise ValueError("NPZ file must contain 'img' and 'labels' keys")

            volume = data["img"]
            mask = data["labels"]

            self.logger.info(
                f"Volume shape: {volume.shape}, Objects: {len(np.unique(mask))-1}"
            )

            # Extract center of mass
            centroid_data = self.extract_center_of_mass(mask)

            # Create visualization if requested
            if save_visualization:
                output_dir = Path(file_path).parent / f"{Path(file_path).stem}_analysis"
                output_dir.mkdir(exist_ok=True)

                viz_path = output_dir / "comprehensive_analysis.png"
                self.create_comprehensive_visualization(
                    volume, mask, centroid_data, str(viz_path)
                )

                # Save centroid data
                results_path = output_dir / "centroid_results.npz"
                np.savez(
                    results_path,
                    centroid=centroid_data["centroid"],
                    bounding_box=centroid_data["bounding_box"],
                    total_volume=centroid_data["total_volume"],
                    num_objects=centroid_data["num_objects"],
                )

                self.logger.info(f"Results saved to: {results_path}")

            return {
                "file_path": file_path,
                "centroid_data": centroid_data,
                "volume_shape": volume.shape,
                "mask_shape": mask.shape,
            }

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
