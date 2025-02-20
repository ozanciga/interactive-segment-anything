from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.events import Tap
import panel as pn
from bokeh_utils import image_to_data_uri
import numpy as np


pn.extension()


class TapSelector:
    def __init__(self, raw_image, resize=1, verbose=False):
        self.verbose = verbose
        self.image_width, self.image_height = raw_image.size
        self.image_url = image_to_data_uri(raw_image)
        self.plot_width, self.plot_height = int(resize * self.image_width), int(resize * self.image_height)
        self.tap_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.tap_display = pn.widgets.StaticText(value="")
        self.tap_plot = self.create_tap_plot()
        self.layout = pn.Column(pn.pane.Bokeh(self.tap_plot), self.tap_display)

    def create_tap_plot(self):
        plot = figure(x_range=(0, self.plot_width), y_range=(0, self.plot_height), title="Tap Tool")
        plot.image_url(url=[self.image_url], x=0, y=self.plot_height, w=self.plot_width, h=self.plot_height)
        plot.scatter('x', 'y', source=self.tap_source, size=10, color='red')
        plot.on_event(Tap, self.tap_callback)
        plot.on_event('reset', self.reset_callback)
        return plot

    def reset_callback(self, event):
        self.tap_source.data = dict(x=[], y=[])  # Clear tapped points

    def tap_callback(self, event):
        self.tap_source.stream({'x': [event.x], 'y': [event.y]})
        if self.verbose:
            self.tap_display.value = f"Tapped at: ({event.x:.2f}, {event.y:.2f})"

    def get_tap_coordinates(self, as_list=False):
        """
        Returns the list of tapped coordinates as (x, y) tuples.
        """
        xy = np.asarray(list(zip(self.tap_source.data["x"], self.tap_source.data["y"])))
        xy[:, 1] = self.plot_height - xy[:, 1]
        xy = (np.asarray(xy) * np.asarray([self.image_width / self.plot_width, self.image_height / self.plot_height])[None]).astype(int)
        
        if as_list:
            return xy.tolist()

        return xy
