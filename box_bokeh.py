from bokeh.models import ColumnDataSource, BoxEditTool
from bokeh.plotting import figure
from PIL import Image
import panel as pn
from bokeh_utils import image_to_data_uri
import numpy as np


pn.extension()


class BoxEditor:
    def __init__(self, raw_image, resize=1, verbose=False):
        self.verbose = verbose
        self.image_width, self.image_height = raw_image.size
        self.image_url = image_to_data_uri(raw_image)
        self.plot_width, self.plot_height = int(resize * self.image_width), int(resize * self.image_height)
        self.box_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], alpha=[]))
        self.box_plot = self.create_boxedit_plot()
        self.layout = pn.pane.Bokeh(self.box_plot)

    def create_boxedit_plot(self):
        plot = figure(x_range=(0, self.plot_width), y_range=(0, self.plot_height), title="BoxEdit Tool")
        plot.image_url(url=[self.image_url], x=0, y=self.plot_height, w=self.plot_width, h=self.plot_height)
        rect = plot.rect(x='x', y='y', width='width', height='height', source=self.box_source, fill_alpha=0.5)
        box_edit_tool = BoxEditTool(renderers=[rect])
        plot.add_tools(box_edit_tool)
        plot.toolbar.active_drag = box_edit_tool
        if self.verbose:
            self.box_source.on_change('data', lambda attr, old, new: print("Updated boxes:", self.box_source.data))
        return plot

    def get_box_coordinates(self, as_list=False):
        """
        Returns the current box coordinates as a list of dictionaries.
        Each dictionary contains x1, y1 [starting leftmost point], x2, y2 [ending rightmost point].
        """
        x_multiplier, y_multiplier = self.image_width / self.plot_width, self.image_height / self.plot_height
        
        box_x1y1x2y2 = []
        
        for x, y, w, h in zip(
            self.box_source.data['x'],
            self.box_source.data['y'],
            self.box_source.data['width'],
            self.box_source.data['height']):
            
            # Calculate box edges (Bokeh's rect glyph is centered)
            left = x - w / 2.0
            right = x + w / 2.0
            bottom = y - h / 2.0
            top = y + h / 2.0

            # Convert data coordinates to pixel indices
            col_start = max(int(np.floor(left * x_multiplier)), 0)
            col_end = min(int(np.ceil(right * x_multiplier)), self.image_width)
            row_start = max(int(np.floor(self.image_height - top * y_multiplier)), 0)
            row_end = min(int(np.ceil(self.image_height - bottom * y_multiplier)), self.image_height)
            
            box_x1y1x2y2.append({'x1': col_start, 'y1': row_start, 'x2': col_end, 'y2': row_end, })
        
        if as_list:
            return [[d[key] for key in d] for d in box_x1y1x2y2]
        
        return box_x1y1x2y2
    
    def compute_box_masks(self):
        """
        Generates binary masks for each drawn box.
        Returns a list of NumPy arrays representing the masks.
        """
        masks = []
        
        box_x1y1x2y2 = self.get_box_coordinates()
        
        for it in box_x1y1x2y2:
            
            mask = np.zeros((self.image_height, self.image_width), dtype=bool)
            mask[it['y1']:it['y2'], it['x1']:it['x2']] = True
            masks.append(mask)

        return masks
