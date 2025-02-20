import panel as pn
from box_bokeh import BoxEditor
from mask_contour_bokeh import MaskContourVisualizer
from tap_bokeh import TapSelector


pn.extension()


class MultiToolVisualizer:
    def __init__(self, raw_image, masks, resize=1, port=5006, verbose=False):
        self.port = port
        self.mask_visualizer = MaskContourVisualizer(raw_image, masks, verbose)
        self.box_editor = BoxEditor(raw_image, resize, verbose)
        self.tap_selector = TapSelector(raw_image, resize, verbose)

        self.layout = pn.Tabs(
            ("Box Editor", self.box_editor.layout),
            ("Tap Selector", self.tap_selector.layout),
            ("Mask Contour Visualization", self.mask_visualizer.layout),
        )

    def serve(self):
        self.stop()
        pn.serve(self.layout, show=True, port=self.port, threaded=True)

    def stop(self):
        for server in pn.state._servers.values():
            try:
                server[0].stop()
            except Exception:
                pass
