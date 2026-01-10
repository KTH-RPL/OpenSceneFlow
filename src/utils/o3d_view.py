"""
Open3D Visualizer for Scene Flow
================================
@date: 2023-1-26 16:38
@author: Qingwen Zhang (https://kin-zhang.github.io/), Ajinkya Khoche (https://ajinkyakhoche.github.io/)
Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology

# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

Features:
  - Single or multi-window visualization
  - Viewpoint sync across windows (press S)
  - Forward/backward playback
  - Screenshot and viewpoint save

CHANGELOG:
2026-01-10 (Qingwen): Unified single/multi visualizer, added viewpoint sync with S key
2024-09-10 (Ajinkya): Add multi-window support, forward/backward playback
2024-08-23 (Qingwen): Use open3d>=0.18.0 set_view_status API
"""

import open3d as o3d
import os
import time
from typing import List, Callable, Union
from functools import partial
import numpy as np


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


COLOR_MAP_HEX = [
    '#a6cee3', '#de2d26', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
    '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#8dd3c7',
    '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
    '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'
]
color_map = [hex_to_rgb(c) for c in COLOR_MAP_HEX]


class O3DVisualizer:
    """
    Unified Open3D visualizer supporting single or multiple windows.
    
    Args:
        view_file: Path to JSON file with saved viewpoint
        res_name: Single name or list of names for windows
        save_folder: Folder to save screenshots
        point_size: Point size for rendering
        bg_color: Background color as RGB tuple (0-1 range)
        screen_width: Screen width for multi-window layout
        screen_height: Screen height for multi-window layout
    
    Usage:
        # Single window
        viz = O3DVisualizer(res_name="flow")
        
        # Multiple windows  
        viz = O3DVisualizer(res_name=["flow", "flow_est"])
    """
    
    def __init__(
        self,
        view_file: str = None,
        res_name: Union[str, List[str]] = "flow",
        save_folder: str = "logs/imgs",
        point_size: float = 3.0,
        bg_color: tuple = (80/255, 90/255, 110/255),
        screen_width: int = 1375,
        screen_height: int = 2500,
    ):
        # Normalize res_name to list
        self.res_names = [res_name] if isinstance(res_name, str) else list(res_name)
        self.num_windows = len(self.res_names)
        
        self.view_file = view_file
        self.save_folder = save_folder
        self.point_size = point_size
        self.bg_color = np.asarray(bg_color)
        
        os.makedirs(self.save_folder, exist_ok=True)
        
        # State
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.playback_direction = 1  # 1: forward, -1: backward
        self.curr_index = -1
        self.tmp_value = None
        self._should_save = False
        self._should_sync = False
        self._sync_source_idx = 0
        
        # Create windows
        self.vis: List[o3d.visualization.VisualizerWithKeyCallback] = []
        self._create_windows(screen_width, screen_height)
        self._setup_render_options()
        self._register_callbacks()
        self._print_help()

    def _create_windows(self, screen_width: int, screen_height: int):
        """Create visualizer windows."""
        if self.num_windows == 1:
            v = o3d.visualization.VisualizerWithKeyCallback()
            title = self._window_title(self.res_names[0])
            v.create_window(window_name=title)
            self.vis.append(v)
        else:
            window_width = screen_width // 2
            window_height = screen_height // 4
            epsilon = 150
            positions = [
                (0, 0),
                (screen_width - window_width + epsilon, 0),
                (0, screen_height - window_height + epsilon),
                (screen_width - window_width + epsilon, screen_height - window_height + epsilon),
            ]
            for i, name in enumerate(self.res_names):
                v = o3d.visualization.VisualizerWithKeyCallback()
                title = self._window_title(name)
                pos = positions[i % len(positions)]
                v.create_window(window_name=title, width=window_width, height=window_height,
                               left=pos[0], top=pos[1])
                self.vis.append(v)

    def _window_title(self, name: str) -> str:
        label = "ground truth flow" if name == "flow" else name
        return f"View {label} | SPACE: play/pause"

    def _setup_render_options(self):
        """Configure render options for all windows."""
        for v in self.vis:
            opt = v.get_render_option()
            opt.background_color = self.bg_color
            opt.point_size = self.point_size

    def _register_callbacks(self):
        """Register keyboard callbacks for all windows."""
        callbacks = [
            (["Ā", "Q", "\x1b"], self._quit),
            ([" "], self._start_stop),
            (["D"], self._next_frame),
            (["A"], self._prev_frame),
            (["P"], self._save_screen),
            (["E"], self._save_error_bar),
            (["S"], self._sync_viewpoint),
        ]
        for keys, callback in callbacks:
            for key in keys:
                for idx, v in enumerate(self.vis):
                    v.register_key_callback(ord(str(key)), partial(callback, src_idx=idx))

    def _print_help(self):
        sync_hint = "[S] sync viewpoint across windows\n" if self.num_windows > 1 else ""
        print(
            f"\nVisualizer initialized ({self.num_windows} window(s)). Keys:\n"
            f"  [SPACE] play/pause    [D] next frame    [A] prev frame\n"
            f"  [P] save screenshot   [E] save error bar\n"
            f"  {sync_hint}"
            f"  [ESC/Q] quit\n"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def update(self, assets: Union[List, List[List]], index: int = -1, value: float = None):
        """
        Update visualizer with new assets.
        
        Args:
            assets: For single window - list of geometries
                    For multi window - list of lists of geometries
            index: Current frame index (for screenshot naming)
            value: Optional value (e.g., max error for colorbar)
        """
        self.curr_index = index
        self.tmp_value = value
        
        # Normalize to list of lists
        if self.num_windows == 1:
            assets_list = [assets] if not self._is_nested_list(assets) else assets
        else:
            assets_list = assets
        
        # Clear and add geometries
        for v in self.vis:
            v.clear_geometries()
        
        for i, window_assets in enumerate(assets_list):
            if i >= len(self.vis):
                break
            for asset in window_assets:
                self.vis[i].add_geometry(asset, reset_bounding_box=False)
                self.vis[i].update_geometry(asset)
        
        # Reset view on first frame
        if self.reset_bounding_box:
            for v in self.vis:
                v.reset_view_point(True)
                if self.view_file is not None:
                    v.set_view_status(open(self.view_file).read())
            self.reset_bounding_box = False
        
        # Render and wait
        for v in self.vis:
            v.update_renderer()
        
        while self.block_vis:
            for v in self.vis:
                v.poll_events()
            if self._should_sync:
                self._do_sync_viewpoint()
            if self._should_save:
                self._do_save_screen()
            if self.play_crun:
                break
        
        self.block_vis = not self.block_vis

    def show(self, assets: List):
        """Show assets and run visualization loop (blocking)."""
        for v in self.vis:
            v.clear_geometries()
        for asset in assets:
            for v in self.vis:
                v.add_geometry(asset)
                if self.view_file is not None:
                    v.set_view_status(open(self.view_file).read())
        for v in self.vis:
            v.update_renderer()
            v.poll_events()
        self.vis[0].run()
        for v in self.vis:
            v.destroy_window()

    def _is_nested_list(self, obj) -> bool:
        """Check if obj is a list of lists."""
        return isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def _quit(self, vis, src_idx=0):
        print("Destroying Visualizer. Thanks for using ^v^.")
        for v in self.vis:
            v.destroy_window()
        os._exit(0)

    def _start_stop(self, vis, src_idx=0):
        self.play_crun = not self.play_crun

    def _next_frame(self, vis, src_idx=0):
        self.block_vis = not self.block_vis
        self.playback_direction = 1

    def _prev_frame(self, vis, src_idx=0):
        self.block_vis = not self.block_vis
        self.playback_direction = -1

    def _save_screen(self, vis, src_idx=0):
        self._should_save = True
        # NOTE: sync viewpoint before saving
        self._should_sync = True
        return False

    def _sync_viewpoint(self, vis, src_idx=0):
        """Sync all windows to the viewpoint of the source window."""
        self._should_sync = True
        self._sync_source_idx = src_idx
        return False

    def _do_sync_viewpoint(self):
        """Actually perform viewpoint sync (called from main loop)."""
        if self.num_windows <= 1:
            self._should_sync = False
            return
        
        source_view = self.vis[self._sync_source_idx].get_view_status()
        for i, v in enumerate(self.vis):
            if i != self._sync_source_idx:
                v.set_view_status(source_view)
                v.update_renderer()
        print(f"Synced viewpoint from window {self._sync_source_idx} to all windows.")
        self._should_sync = False

    def _do_save_screen(self):
        """Save screenshots from all windows."""
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        
        for i, v in enumerate(self.vis):
            v.poll_events()
            v.update_renderer()
            
            name = self.res_names[i] if i < len(self.res_names) else f"window{i}"
            prefix = f"{self.curr_index}_{name}" if self.curr_index != -1 else name
            png_file = f"{self.save_folder}/{prefix}_{timestamp}.png"
            v.capture_screen_image(png_file)
            
            if i == 0:
                view_file = f"{self.save_folder}/{prefix}_{timestamp}.json"
                with open(view_file, 'w') as f:
                    f.write(v.get_view_status())
        
        print(f"Screenshots saved to {self.save_folder}/")
        self._should_save = False

    def _save_error_bar(self, vis, src_idx=0):
        """Save error colorbar as image."""
        if self.tmp_value is None:
            print("No error value set, skipping error bar save.")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        prefix = f"{self.curr_index}_error" if self.curr_index != -1 else "error"
        png_file = f"{self.save_folder}/{prefix}_{timestamp}.png"
        
        fig, ax = plt.subplots(figsize=(10, 1))
        max_val = self.tmp_value * 100
        norm = mpl.colors.Normalize(vmin=0, vmax=max_val)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.cm.hot, norm=norm, orientation='horizontal')
        
        ticks = np.linspace(0, max_val, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}" for t in ticks])
        cb.set_label('Error Magnitude (cm)')
        
        plt.savefig(png_file, bbox_inches='tight')
        plt.close()
        print(f"Error bar saved to: {png_file}")


# Backward compatibility aliases; FIXME: remove in near future
MyVisualizer = O3DVisualizer
MyMultiVisualizer = O3DVisualizer


if __name__ == "__main__":
    json_content = """{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 3.9660897254943848, 2.427476167678833, 2.55859375 ],
			"boundingbox_min" : [ 0.55859375, 0.83203125, 0.56663715839385986 ],
			"field_of_view" : 60.0,
			"front" : [ 0.27236083595988803, -0.25567329763523589, -0.92760484038816615 ],
			"lookat" : [ 2.4114965637897101, 1.8070288935660688, 1.5662280268112718 ],
			"up" : [ -0.072779625398507866, -0.96676294585190281, 0.24509698622097265 ],
			"zoom" : 0.47999999999999976
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""
    view_json_file = "view.json"
    with open(view_json_file, 'w') as f:
        f.write(json_content)
    
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    
    viz = O3DVisualizer(view_json_file, res_name="Demo")
    viz.show([pcd])