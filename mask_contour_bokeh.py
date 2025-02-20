import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.events import Tap
from skimage import measure
from matplotlib.path import Path


pn.extension()


class MaskContourVisualizer:
    def __init__(self, raw_image, masks, verbose=False):
        self.verbose = verbose
        self.image = np.asarray(raw_image)
        self.masks = masks
        self.height, self.width, _ = self.image.shape
        self.source = self._generate_contour_data()
        self.fixed_mask_indices = set()
        self.selected_text = pn.widgets.StaticText(value="")
        self.plot = self._setup_plot()
        self.layout = pn.Column(self.plot, self.selected_text)

    def _generate_contour_data(self):
        xs_list, ys_list, mask_index_list = [], [], []
        for idx, mask in enumerate(self.masks):
            contours = measure.find_contours(mask, level=0.5)
            for contour in contours:
                xs = contour[:, 1].tolist()  # x = col, y = row
                ys = (self.height - contour[:, 0]).tolist()  # Flip y-axis
                xs_list.append(xs)
                ys_list.append(ys)
                mask_index_list.append(idx)
        return ColumnDataSource(data=dict(xs=xs_list, ys=ys_list, mask_index=mask_index_list))

    def _setup_plot(self):
        p = figure(x_range=(0, self.width), y_range=(0, self.height), title="Mask Contour Visualization")
        p.image_rgba(image=[self._rgb_to_rgba(as_2d=True)], x=0, y=0, dw=self.width, dh=self.height)
        
        # Overlay masks as patches
        patches_renderer = p.patches(
            'xs', 'ys', source=self.source,
            fill_color="magenta", line_color="black",
            fill_alpha=0.05, line_alpha=0.05
        )
        # Hover & selection mask style.
        patches_renderer.hover_glyph = type(patches_renderer.glyph).clone(patches_renderer.glyph)
        patches_renderer.hover_glyph.fill_alpha = 0.35
        patches_renderer.hover_glyph.line_alpha = 0.35

        patches_renderer.selection_glyph = type(patches_renderer.glyph).clone(patches_renderer.glyph)
        patches_renderer.selection_glyph.fill_alpha = 0.50
        patches_renderer.selection_glyph.line_alpha = 0.50

        p.add_tools(HoverTool(renderers=[patches_renderer], tooltips=None))
        p.on_event(Tap, self._tap_callback)
        return p

    def _rgb_to_rgba(self, as_2d=False):
        """
        Convert an RGB image to an RGBA image with full opacity.
        as_2d: bokeh rgba expects the image as 2d
        
        :return: RGBA image as a NumPy array.
        """
        h, w, _ = self.image.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = self.image
        rgba[..., 3] = 255  # Full opacity
        rgba = rgba[::-1]  # Flip vertically
        if as_2d:
            rgba = rgba.view(dtype=np.uint32).reshape((self.height, self.width))
        return rgba

    def _point_in_polygon(self, x, y, xs, ys):
        """
        Check if a given point is inside a polygon defined by xs, ys.
        
        :param x: X coordinate of the point.
        :param y: Y coordinate of the point.
        :param xs: X coordinates of the polygon.
        :param ys: Y coordinates of the polygon.
        :return: True if the point is inside the polygon, else False.
        """
        poly = Path(list(zip(xs, ys)))
        return poly.contains_point((x, y))

    def _tap_callback(self, event):
        data = self.source.data
        hit_mask_indices = [data['mask_index'][i] for i, (xs, ys) in enumerate(zip(data['xs'], data['ys']))
                            if Path(list(zip(xs, ys))).contains_point((event.x, event.y))]
        if hit_mask_indices:
            clicked_mask = hit_mask_indices[0]
            self.fixed_mask_indices ^= {clicked_mask}
            self.source.selected.indices = [i for i, m in enumerate(data['mask_index']) if m in self.fixed_mask_indices]
            if self.verbose:
                self.selected_text.value = f"Fixed mask indices: {sorted(self.fixed_mask_indices)}"

