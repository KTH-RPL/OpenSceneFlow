'''
# @date: 2023-1-26 16:38
# @author: Qingwen Zhang  (https://kin-zhang.github.io/), Ajinkya Khoche (https://ajinkyakhoche.github.io/)
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# @detail:
#  1. Play the data you want in open3d, and save the view control to json file.
#  2. Use the json file to view the data again.
#  3. Save the screen shot and view file for later check and animation.
# 
# code gits: https://gist.github.com/Kin-Zhang/77e8aa77a998f1a4f7495357843f24ef
# 
# CHANGELOG:
# 2024-08-23 21:41(Qingwen): remove totally on view setting from scratch but use open3d>=0.18.0 version for set_view from json text func.
# 2024-04-15 12:06(Qingwen): show a example json text. add hex_to_rgb, color_map_hex, color_map (for color points if needed)
# 2024-01-27 0:41(Qingwen): update MyVisualizer class, reference from kiss-icp: https://github.com/PRBonn/kiss-icp/blob/main/python/kiss_icp/tools/visualizer.py
# 2024-09-10 (Ajinkya): Add MyMultiVisualizer class to view multiple windows at once, allow forward and backward playback, create bev square for giving a sense of metric scale.
'''

import open3d as o3d
import os, time
from typing import List, Callable
from functools import partial
import numpy as np

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

color_map_hex = ['#a6cee3', '#de2d26', '#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00',\
                 '#cab2d6','#6a3d9a','#ffff99','#b15928', '#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3',\
                 '#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

color_map = [hex_to_rgb(color) for color in color_map_hex]

        
class MyVisualizer:
    def __init__(self, view_file=None, window_title="Default", save_folder="logs/imgs"):
        self.params = None
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_title)
        self.view_file = view_file

        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.save_img_folder = save_folder
        os.makedirs(self.save_img_folder, exist_ok=True)
        print(
            f"\n{window_title.capitalize()} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t[ESC/Q] to exit\n"
            "\t    [P] to save screen and viewpoint\n"
            "\t    [D] to step next\n"
        )
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback(["P"], self._save_screen)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["D"], self._next_frame)

    def show(self, assets: List):
        self.vis.clear_geometries()

        for asset in assets:
            self.vis.add_geometry(asset)
            if self.view_file is not None:
                self.vis.set_view_status(open(self.view_file).read())

        self.vis.update_renderer()
        self.vis.poll_events()
        self.vis.run()
        self.vis.destroy_window()

    def update(self, assets: List, clear: bool = True):
        if clear:
            self.vis.clear_geometries()

        for asset in assets:
            self.vis.add_geometry(asset, reset_bounding_box=False)
            self.vis.update_geometry(asset)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            if self.view_file is not None:
                self.vis.set_view_status(open(self.view_file).read())
            self.reset_bounding_box = False

        self.vis.update_renderer()
        while self.block_vis:
            self.vis.poll_events()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _quit(self, vis):
        print("Destroying Visualizer. Thanks for using ^v^.")
        vis.destroy_window()
        os._exit(0)

    def _save_screen(self, vis):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        png_file = f"{self.save_img_folder}/ScreenShot_{timestamp}.png"
        view_json_file = f"{self.save_img_folder}/ScreenView_{timestamp}.json"
        with open(view_json_file, 'w') as f:
            f.write(vis.get_view_status())
        vis.capture_screen_image(png_file)
        print(f"ScreenShot saved to: {png_file}, Please check it.")


def create_bev_square(size=409.6, color=[68/255,114/255,196/255]):
    # Create the vertices of the square
    half_size = size / 2.0
    vertices = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ])

    # Define the square as a LineSet for visualization
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [color for _ in lines]  

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

class MyMultiVisualizer(MyVisualizer):
    def __init__(self, view_file=None, flow_mode=['flow'], screen_width=2500, screen_height = 1375):
        self.params = None
        self.view_file = view_file
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.playback_direction = 1 # 1:forward, -1:backward

        self.vis = []
        # self.o3d_vctrl = []

        # Define width and height for each window
        window_width = screen_width // 2
        window_height = screen_height // 2
        # Define positions for the four windows
        epsilon = 150
        positions = [
            (0, 0),  # Top-left
            (screen_width - window_width + epsilon, 0),  # Top-right
            (0, screen_height - window_height + epsilon),  # Bottom-left
            (screen_width - window_width + epsilon, screen_height - window_height + epsilon)  # Bottom-right
        ]

        for i, mode in enumerate(flow_mode):
            window_title = f"view {'ground truth flow' if mode == 'flow' else f'{mode} flow'}, `SPACE` start/stop"
            v = o3d.visualization.VisualizerWithKeyCallback()
            v.create_window(window_name=window_title, width=window_width, height=window_height, left=positions[i%len(positions)][0], top=positions[i%len(positions)][1])
            # self.o3d_vctrl.append(ViewControl(v.get_view_control(), view_file=view_file))
            self.vis.append(v)

        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["D"], self._next_frame)
        self._register_key_callback(["A"], self._prev_frame)
        print(
            f"\n{window_title.capitalize()} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t[ESC/Q] to exit\n"
            "\t    [P] to save screen and viewpoint\n"
            "\t    [D] to step next\n"
            "\t    [A] to step previous\n"
        )

    def update(self, assets_list: List, clear: bool = True):
        if clear:
            [v.clear_geometries() for v in self.vis]

        for i, assets in enumerate(assets_list):
            [self.vis[i].add_geometry(asset, reset_bounding_box=False) for asset in assets]
            self.vis[i].update_geometry(assets[-1])

        if self.reset_bounding_box:
            [v.reset_view_point(True) for v in self.vis]
            if self.view_file is not None:
                # [o.read_viewTfile(self.view_file) for o in self.o3d_vctrl]
                [v.set_view_status(open(self.view_file).read()) for v in self.vis]
            self.reset_bounding_box = False

        [v.update_renderer() for v in self.vis]
        while self.block_vis:
            [v.poll_events() for v in self.vis]
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            [v.register_key_callback(ord(str(key)), partial(callback)) for v in self.vis]
    def _next_frame(self, vis):
        self.block_vis = not self.block_vis
        self.playback_direction = 1
    def _prev_frame(self, vis):
        self.block_vis = not self.block_vis
        self.playback_direction = -1


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
    # write to json file
    view_json_file = "view.json"
    with open(view_json_file, 'w') as f:
        f.write(json_content)
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)

    viz = MyVisualizer(view_json_file, window_title="Qingwen's View")
    viz.show([pcd])