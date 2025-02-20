
from base64 import b64encode
from io import BytesIO
from PIL import Image


def image_to_data_uri(image=None, image_path=None):
    """
    Convert a PIL Image to a base64 data URI.
    
    Args:
        image (PIL.Image.Image): The image to convert.
    
    Returns:
        str: Data URI containing the base64-encoded image.
    """
    if image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded = b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    elif image_path:
        image_path = image
        with open(image_path, "rb") as img_file:
            encoded = b64encode(img_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    else:
        raise NotImplementedError

