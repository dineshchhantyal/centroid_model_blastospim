"""
3D Volume Visualizer

This module provides comprehensive 3D visualization capabilities for volumetric data
with support for anisotropic voxel spacing and additional point overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns


class Volume3DVisualizer:
    """
    3D Volume Visualizer with support for anisotropic data and point overlays.
    """

    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (2.0, 0.208, 0.208),
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the 3D Volume Visualizer.

        Args:
            voxel_size: Physical voxel size (z, y, x) in micrometers
            logger: Optional logger instance
        """
        self.voxel_size = np.array(voxel_size)

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Default visualization parameters
        self.default_params = {
            "figure_size": (12, 10),
            "point_size": 1,
            "point_alpha": 0.3,
            "centroid_size": 100,
            "centroid_color": "red",
            "centroid_marker": "*",
            "colormap": "viridis",
            "background_color": "white",
            "grid": True,
            "max_points": 10000,  # Subsample for performance
        }

    def create_3d_scatter(
        self,
        volume: np.ndarray,
        threshold: float = 0.5,
        extra_points: Optional[Dict[str, Dict[str, Any]]] = None,
        title: str = "3D Volume Visualization",
        physical_coordinates: bool = True,
        subsample: bool = True,
        colormap: str = None,
        figsize: Tuple[float, float] = None,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create a 3D scatter plot of the volume data.

        Args:
            volume: 3D numpy array to visualize
            threshold: Threshold value for displaying voxels (voxels > threshold are shown)
            extra_points: Dictionary of additional points to plot. Format:
                {
                    'point_name': {
                        'coordinates': (z, y, x) or [(z1,y1,x1), (z2,y2,x2), ...],
                        'color': 'red',
                        'size': 100,
                        'marker': '*',
                        'alpha': 1.0,
                        'label': 'Centroid'
                    }
                }
            title: Plot title
            physical_coordinates: If True, convert to physical coordinates using voxel_size
            subsample: If True, subsample points for better performance
            colormap: Colormap name for volume points
            figsize: Figure size tuple
            save_path: Path to save the figure
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib Figure object
        """
        # Set default parameters
        params = {**self.default_params, **kwargs}
        figsize = figsize or params["figure_size"]
        colormap = colormap or params["colormap"]

        # Get coordinates of voxels above threshold
        coords = np.where(volume > threshold)

        if len(coords[0]) == 0:
            self.logger.warning("No voxels found above threshold")
            return None

        # Get intensity values
        intensities = volume[coords]

        # Convert to physical coordinates if requested
        if physical_coordinates:
            print("Using physical coordinates for visualization.")
            print(coords, "coords", ":::", self.voxel_size, "voxel_size")
            z_coords = coords[0] * self.voxel_size[0]
            y_coords = coords[1] * self.voxel_size[1]
            x_coords = coords[2] * self.voxel_size[2]
            coord_unit = "μm"
        else:
            z_coords = coords[0]
            y_coords = coords[1]
            x_coords = coords[2]
            coord_unit = "voxels"

        # Subsample for performance if requested
        if subsample and len(z_coords) > params["max_points"]:
            self.logger.info(
                f"Subsampling {len(z_coords)} points to {params['max_points']} for performance"
            )
            idx = np.random.choice(len(z_coords), params["max_points"], replace=False)
            z_coords = z_coords[idx]
            y_coords = y_coords[idx]
            x_coords = x_coords[idx]
            intensities = intensities[idx]

        # Create figure and 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot volume points
        scatter = ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=intensities,
            cmap=colormap,
            s=params["point_size"],
            alpha=params["point_alpha"],
            label="Volume Data",
        )

        # Add colorbar for volume intensities
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label("Intensity", rotation=270, labelpad=20)

        # Plot extra points if provided
        print("Extra points:", extra_points)
        if extra_points:
            for point_name, point_config in extra_points.items():
                self._add_extra_points(
                    ax, point_config, physical_coordinates, coord_unit, point_name
                )

        # Set labels and title
        ax.set_xlabel(f"X ({coord_unit})")
        ax.set_ylabel(f"Y ({coord_unit})")
        ax.set_zlabel(f"Z ({coord_unit})")
        ax.set_title(title)

        # Set equal aspect ratio considering anisotropic data
        if physical_coordinates:
            self._set_equal_aspect_3d(ax, x_coords, y_coords, z_coords)

        # Add grid if requested
        if params["grid"]:
            ax.grid(True, alpha=0.3)

        # Add legend if there are extra points
        if extra_points:
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

        # Add information text
        info_text = self._create_info_text(
            volume, len(z_coords), threshold, physical_coordinates
        )
        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"3D visualization saved to: {save_path}")

        return fig

    def create_multi_threshold_view(
        self,
        volume: np.ndarray,
        thresholds: List[float] = [0.3, 0.5, 0.7],
        extra_points: Optional[Dict[str, Dict[str, Any]]] = None,
        physical_coordinates: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create multiple 3D views with different thresholds.

        Args:
            volume: 3D numpy array to visualize
            thresholds: List of threshold values to display
            extra_points: Additional points to overlay
            physical_coordinates: If True, use physical coordinates
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        n_views = len(thresholds)
        fig = plt.figure(figsize=(6 * n_views, 6))

        coord_unit = "μm" if physical_coordinates else "voxels"

        for i, threshold in enumerate(thresholds):
            ax = fig.add_subplot(1, n_views, i + 1, projection="3d")

            # Get coordinates for this threshold
            coords = np.where(volume > threshold)

            if len(coords[0]) == 0:
                ax.text2D(
                    0.5,
                    0.5,
                    f"No data above\nthreshold {threshold}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue

            intensities = volume[coords]

            # Convert coordinates
            if physical_coordinates:
                z_coords = coords[0] * self.voxel_size[0]
                y_coords = coords[1] * self.voxel_size[1]
                x_coords = coords[2] * self.voxel_size[2]
            else:
                z_coords, y_coords, x_coords = coords

            # Subsample if too many points
            if len(z_coords) > self.default_params["max_points"]:
                idx = np.random.choice(
                    len(z_coords), self.default_params["max_points"], replace=False
                )
                z_coords = z_coords[idx]
                y_coords = y_coords[idx]
                x_coords = x_coords[idx]
                intensities = intensities[idx]

            # Plot volume points
            scatter = ax.scatter(
                x_coords,
                y_coords,
                z_coords,
                c=intensities,
                cmap="viridis",
                s=1,
                alpha=0.4,
            )

            # Add extra points
            if extra_points:
                for point_name, point_config in extra_points.items():
                    self._add_extra_points(
                        ax, point_config, physical_coordinates, coord_unit, point_name
                    )

            ax.set_title(f"Threshold: {threshold}")
            ax.set_xlabel(f"X ({coord_unit})")
            ax.set_ylabel(f"Y ({coord_unit})")
            ax.set_zlabel(f"Z ({coord_unit})")

            if physical_coordinates:
                self._set_equal_aspect_3d(ax, x_coords, y_coords, z_coords)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Multi-threshold visualization saved to: {save_path}")

        return fig

    def create_slice_overlay_3d(
        self,
        volume: np.ndarray,
        slice_indices: Optional[Dict[str, int]] = None,
        extra_points: Optional[Dict[str, Dict[str, Any]]] = None,
        threshold: float = 0.5,
        physical_coordinates: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create 3D visualization with 2D slice overlays.

        Args:
            volume: 3D numpy array to visualize
            slice_indices: Dictionary with slice indices {'z': idx, 'y': idx, 'x': idx}
            extra_points: Additional points to overlay
            threshold: Threshold for volume points
            physical_coordinates: If True, use physical coordinates
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Default to middle slices
        if slice_indices is None:
            slice_indices = {
                "z": volume.shape[0] // 2,
                "y": volume.shape[1] // 2,
                "x": volume.shape[2] // 2,
            }

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        coord_unit = "μm" if physical_coordinates else "voxels"

        # Plot volume points
        coords = np.where(volume > threshold)
        if len(coords[0]) > 0:
            intensities = volume[coords]

            if physical_coordinates:
                z_coords = coords[0] * self.voxel_size[0]
                y_coords = coords[1] * self.voxel_size[1]
                x_coords = coords[2] * self.voxel_size[2]
            else:
                z_coords, y_coords, x_coords = coords

            # Subsample if needed
            if len(z_coords) > self.default_params["max_points"]:
                idx = np.random.choice(
                    len(z_coords), self.default_params["max_points"], replace=False
                )
                z_coords = z_coords[idx]
                y_coords = y_coords[idx]
                x_coords = x_coords[idx]
                intensities = intensities[idx]

            ax.scatter(
                x_coords,
                y_coords,
                z_coords,
                c=intensities,
                cmap="viridis",
                s=0.5,
                alpha=0.2,
                label="Volume Data",
            )

        # Add slice planes
        self._add_slice_planes(ax, volume, slice_indices, physical_coordinates)

        # Add extra points
        if extra_points:
            for point_name, point_config in extra_points.items():
                self._add_extra_points(
                    ax, point_config, physical_coordinates, coord_unit, point_name
                )

        ax.set_xlabel(f"X ({coord_unit})")
        ax.set_ylabel(f"Y ({coord_unit})")
        ax.set_zlabel(f"Z ({coord_unit})")
        ax.set_title("3D Volume with Slice Overlays")

        if physical_coordinates and len(coords[0]) > 0:
            self._set_equal_aspect_3d(ax, x_coords, y_coords, z_coords)

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Slice overlay visualization saved to: {save_path}")

        return fig

    def _add_extra_points(
        self,
        ax: plt.Axes,
        point_config: Dict[str, Any],
        physical_coordinates: bool,
        coord_unit: str,
        point_name: str,
    ) -> None:
        """Add extra points to the 3D plot."""
        coordinates = point_config["coordinates"]

        # Handle both single point and multiple points
        if isinstance(coordinates[0], (int, float)):
            # Single point: (z, y, x)
            coordinates = [coordinates]

        # Convert coordinates if needed
        plot_coords = []
        for coord in coordinates:
            if physical_coordinates:
                z_phys = coord[0] * self.voxel_size[0]
                y_phys = coord[1] * self.voxel_size[1]
                x_phys = coord[2] * self.voxel_size[2]
                plot_coords.append((x_phys, y_phys, z_phys))
            else:
                plot_coords.append(
                    (coord[2], coord[1], coord[0])
                )  # x, y, z for plotting

        # Extract x, y, z coordinates
        x_coords = [coord[0] for coord in plot_coords]
        y_coords = [coord[1] for coord in plot_coords]
        z_coords = [coord[2] for coord in plot_coords]

        # Plot points
        ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=point_config.get("color", "red"),
            s=point_config.get("size", 100),
            marker=point_config.get("marker", "*"),
            alpha=point_config.get("alpha", 1.0),
            label=point_config.get("label", point_name),
            edgecolors="black",
            linewidth=1,
        )

        # Add text labels if specified
        if point_config.get("add_labels", False):
            for i, (x, y, z) in enumerate(plot_coords):
                label_text = f"{point_name}_{i}" if len(plot_coords) > 1 else point_name
                ax.text(x, y, z, f"  {label_text}", fontsize=8)

    def _add_slice_planes(
        self,
        ax: plt.Axes,
        volume: np.ndarray,
        slice_indices: Dict[str, int],
        physical_coordinates: bool,
    ) -> None:
        """Add 2D slice planes to the 3D plot."""
        if physical_coordinates:
            z_range = np.arange(volume.shape[0]) * self.voxel_size[0]
            y_range = np.arange(volume.shape[1]) * self.voxel_size[1]
            x_range = np.arange(volume.shape[2]) * self.voxel_size[2]
        else:
            z_range = np.arange(volume.shape[0])
            y_range = np.arange(volume.shape[1])
            x_range = np.arange(volume.shape[2])

        # Z slice (XY plane)
        if "z" in slice_indices:
            z_idx = slice_indices["z"]
            z_slice = volume[z_idx, :, :]
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, z_range[z_idx])
            ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=plt.cm.gray(z_slice / z_slice.max()),
                alpha=0.3,
                antialiased=False,
            )

        # Y slice (XZ plane)
        if "y" in slice_indices:
            y_idx = slice_indices["y"]
            y_slice = volume[:, y_idx, :]
            X, Z = np.meshgrid(x_range, z_range)
            Y = np.full_like(X, y_range[y_idx])
            ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=plt.cm.gray(y_slice / y_slice.max()),
                alpha=0.3,
                antialiased=False,
            )

        # X slice (YZ plane)
        if "x" in slice_indices:
            x_idx = slice_indices["x"]
            x_slice = volume[:, :, x_idx]
            Y, Z = np.meshgrid(y_range, z_range)
            X = np.full_like(Y, x_range[x_idx])
            ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=plt.cm.gray(x_slice / x_slice.max()),
                alpha=0.3,
                antialiased=False,
            )

    def _set_equal_aspect_3d(
        self,
        ax: plt.Axes,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray,
    ) -> None:
        """Set equal aspect ratio for 3D plot considering anisotropic data."""
        # Get data ranges
        x_range = np.ptp(x_coords)
        y_range = np.ptp(y_coords)
        z_range = np.ptp(z_coords)

        # Get centers
        x_center = (np.max(x_coords) + np.min(x_coords)) / 2
        y_center = (np.max(y_coords) + np.min(y_coords)) / 2
        z_center = (np.max(z_coords) + np.min(z_coords)) / 2

        # Use the maximum range for all axes
        max_range = max(x_range, y_range, z_range) / 2

        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)

    def _create_info_text(
        self,
        volume: np.ndarray,
        n_points: int,
        threshold: float,
        physical_coordinates: bool,
    ) -> str:
        """Create information text for the plot."""
        coord_unit = "μm" if physical_coordinates else "voxels"

        if physical_coordinates:
            size_text = (
                f"{volume.shape[0]*self.voxel_size[0]:.1f} × "
                f"{volume.shape[1]*self.voxel_size[1]:.1f} × "
                f"{volume.shape[2]*self.voxel_size[2]:.1f} {coord_unit}³"
            )
        else:
            size_text = f"{volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]} {coord_unit}"

        return (
            f"Volume: {size_text}\n"
            f"Voxel size: {self.voxel_size[0]:.3f}, {self.voxel_size[1]:.3f}, {self.voxel_size[2]:.3f} μm\n"
            f"Threshold: {threshold}\n"
            f"Points shown: {n_points:,}"
        )

    def create_centroid_comparison(
        self,
        volume: np.ndarray,
        centroids: Dict[str, Tuple[float, float, float]],
        threshold: float = 0.5,
        physical_coordinates: bool = True,
        save_path: Optional[str] = None,
        extra_points: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> plt.Figure:
        """
        Create 3D visualization comparing multiple centroid methods.

        Args:
            volume: 3D numpy array to visualize
            centroids: Dictionary of centroid methods and their coordinates
                      {'method_name': (z, y, x), ...}
            threshold: Threshold for volume visualization
            physical_coordinates: If True, use physical coordinates
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Convert centroids to extra_points format
        colors = ["red", "blue", "green", "orange", "purple", "brown"]
        markers = ["*", "o", "^", "s", "D", "v"]

        for i, (method, coords) in enumerate(centroids.items()):
            extra_points[method] = {
                "coordinates": coords,
                "color": colors[i % len(colors)],
                "size": 150,
                "marker": markers[i % len(markers)],
                "alpha": 0.9,
                "label": method,
                "add_labels": True,
            }

        return self.create_3d_scatter(
            volume=volume,
            threshold=threshold,
            extra_points=extra_points,
            title="Centroid Method Comparison",
            physical_coordinates=physical_coordinates,
            save_path=save_path,
        )


# Example usage and convenience functions
def visualize_volume_with_centroid(
    volume: np.ndarray,
    centroid: Tuple[float, float, float],
    voxel_size: Tuple[float, float, float] = (2.0, 0.208, 0.208),
    threshold: float = 0.5,
    title: str = "3D Volume with Centroid",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Convenience function to visualize volume with a single centroid.

    Args:
        volume: 3D numpy array
        centroid: Centroid coordinates (z, y, x)
        voxel_size: Voxel size (z, y, x) in micrometers
        threshold: Threshold for volume visualization
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    visualizer = Volume3DVisualizer(voxel_size=voxel_size)

    extra_points = {
        "centroid": {
            "coordinates": centroid,
            "color": "red",
            "size": 200,
            "marker": "*",
            "alpha": 1.0,
            "label": "Centroid",
        }
    }

    return visualizer.create_3d_scatter(
        volume=volume,
        threshold=threshold,
        extra_points=extra_points,
        title=title,
        save_path=save_path,
    )


def visualize_multiple_objects(
    volume: np.ndarray,
    object_centroids: List[Tuple[float, float, float]],
    voxel_size: Tuple[float, float, float] = (2.0, 0.208, 0.208),
    threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Convenience function to visualize volume with multiple object centroids.

    Args:
        volume: 3D numpy array
        object_centroids: List of centroid coordinates [(z, y, x), ...]
        voxel_size: Voxel size (z, y, x) in micrometers
        threshold: Threshold for volume visualization
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    visualizer = Volume3DVisualizer(voxel_size=voxel_size)

    extra_points = {
        "object_centroids": {
            "coordinates": object_centroids,
            "color": "red",
            "size": 100,
            "marker": "o",
            "alpha": 0.8,
            "label": f"{len(object_centroids)} Objects",
            "add_labels": True,
        }
    }

    return visualizer.create_3d_scatter(
        volume=volume,
        threshold=threshold,
        extra_points=extra_points,
        title=f"3D Volume with {len(object_centroids)} Object Centroids",
        save_path=save_path,
    )
