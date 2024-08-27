import numpy as np
import cv2



def convert_to_uint8(image):  
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def new_bg(data, threshold, penlightness, new_ink_col):
    # threshold = threshold/255
    # mask = (data > threshold).astype(np.float32)
    mask = (data > threshold).astype(np.float32)
    new_ink_color = np.array(new_ink_col, dtype=np.float32)
  
    if penlightness > 1:
        penlightness = penlightness / 10
        # print(penlightness)
        new_ink_color = new_ink_color + (255 - new_ink_color) * penlightness
    else:
        new_ink_color *= penlightness

    color_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)  
    color_data[:] = new_ink_color
    color_data = color_data * mask + data * (1-mask)      

    return color_data


def contrast_image(data, threshold, scale):
    grayscale = np.mean(data, axis=2)
    mask = grayscale < threshold
    # data[mask] = np.minimum(255, data[mask] * scale)
    data[mask] = data[mask] * scale

    return data


def darken_image(data, threshold, scale):
    # threshold = threshold/255
    mask = data < threshold
    data[mask] = data[mask] * scale

    return data

def new_ink(data, threshold, penlightness, new_ink_col, sketch=False):
    data = data/255
    threshold = threshold/255
    # print(data)
    # print(threshold)
    mask = (data < threshold)
    mask2 = (data >= threshold)
    new_ink_color = np.array(new_ink_col, dtype=np.float32)
  
    if penlightness > 1:
        # Lighten the color: scale up towards white (255, 255, 255)
        penlightness = penlightness / 10
        # print(penlightness)
        new_ink_color = new_ink_color + (255 - new_ink_color) * penlightness
    else:
        new_ink_color *= penlightness
    # print(new_ink_color) 
    color_data = data.astype(np.float32)
    color_data2 = data.astype(np.float32)   
    color_data = color_data * (1 - mask) + new_ink_color * data * mask/255     
    color_data2 = color_data2 * (1 - mask2) + new_ink_color * data/255
    color_data = color_data + color_data2 * mask2

    return color_data*255


def new_ink2(data, threshold, penlightness, new_ink_col):
    # threshold = threshold/255
    mask = (data > threshold).astype(np.float32)
    new_ink_color = np.array(new_ink_col, dtype=np.float32)
  
    if penlightness > 1:
        # Lighten the color: scale up towards white (255, 255, 255)
        penlightness = penlightness / 10
        # print(penlightness)
        new_ink_color = new_ink_color + (255 - new_ink_color) * penlightness
    else:
        new_ink_color *= penlightness
    color_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)  
    color_data[:] = new_ink_color
    color_data = color_data + data * mask      

    return color_data

def ensure_grayscale(image):
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        pass  # Already grayscale
    else:
        raise ValueError("Unsupported image format")
    return image


def normalize_image(image):
    # Convert to float if not already
    image = image.astype(np.float32)
    
    # Normalize to range [0, 1]
    image_min = image.min()
    image_max = image.max()
    # print(f"Image max: {image_max }")
    
    # Avoid division by zero
    if image_max > image_min:
        normalized_image = (image - image_min) / (image_max - image_min)
    else:
        normalized_image = np.zeros_like(image)
    # print(f"Image max after normalization: {normalized_image.max() }")

    return normalized_image

def remove_noise_from_sketch(image, kn=2, min_size=40, threshold_value=50):
    sketch = ensure_grayscale(image)
    sketch = convert_to_uint8(sketch)
    kernel_size = (kn, kn)

    sketch_inv = cv2.bitwise_not(sketch)
    
    _, binary_sketch = cv2.threshold(sketch_inv, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones(kernel_size, np.uint8)
    opened_sketch = cv2.morphologyEx(binary_sketch, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(opened_sketch)
    
    cleaned_sketch = np.zeros_like(opened_sketch)
    for label in range(1, num_labels):  # Skip the background label (0)
        if np.sum(labels_im == label) >= min_size:
            cleaned_sketch[labels_im == label] = 255

    return cv2.bitwise_not(cleaned_sketch)


def getColor(col):
    color_dict = {
        "white": [255, 255, 255],
        "black": [0, 0, 0],
        "red": [255, 0, 0],
        "lime": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "cyan": [0, 255, 255],
        "magenta": [255, 0, 255],
        "silver": [192, 192, 192],
        "gray": [128, 128, 128],
        "maroon": [128, 0, 0],
        "olive": [128, 128, 0],
        "green": [0, 128, 0],
        "purple": [128, 0, 128],
        "teal": [0, 128, 128],
        "navy": [0, 0, 128],
    }

    return color_dict.get(col, [0, 0, 0])


