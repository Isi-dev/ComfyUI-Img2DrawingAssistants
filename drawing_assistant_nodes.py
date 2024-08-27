import cv2
import numpy as np
import torch
from .utils import new_bg, darken_image, new_ink, new_ink2, contrast_image, getColor, convert_to_uint8, remove_noise_from_sketch



class Sketch_Assistant_grayScale():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_strength": ("FLOAT", {"default": 1, "min": 1, "max": 10, "step": 0.1}),
                "shading_effect": ("INT", {"default": 41, "min": 5, "max": 105, "step": 2}),
                "details": ("INT", {"default": 225, "min": 50, "max": 255, "step": 1}),
                "smoothness": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 150.0, "step": 1.0}),
                "noise_removal": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),    
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("sketch", "grayscale image")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, image, line_strength, shading_effect, details, smoothness, contrast, noise_removal):
        imgNo =  image.shape[0]
        if imgNo == 1:
            print("1 image received for conversion to grayscale sketch")
        else:
            print(f"{imgNo} images received for conversion to grayscale sketch")
        images = []
        bandwImages = []
        for img in image:
            sketch, bandwI = processImg2SketchGrayScale(img, line_strength, shading_effect, details, smoothness, contrast, noise_removal)
            images.append(sketch)
            bandwImages.append(bandwI)
        
        images = torch.cat(images, dim=0)
        bandwImages = torch.cat(bandwImages, dim=0)
        print("Conversion complete!")

        return (images, bandwImages)   
    

class Sketch_Assistant():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "artist": (["1", "2"],),
                "line_strength": ("FLOAT", {"default": 2, "min": 1, "max": 10, "step": 0.1}),
                "shading_effect": ("INT", {"default": 41, "min": 5, "max": 105, "step": 2}),
                "line_color": (["black", "white", "red", "lime", "blue", "yellow", "cyan", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"],),
                "details": ("INT", {"default": 240, "min": 50, "max": 255, "step": 1}),
                "smoothness": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1}),   
                "enable_bg_color_change": ("BOOLEAN", { "default": False }),
                "bg_color": (["white", "black", "red", "lime", "blue", "yellow", "cyan", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"],),  
                "bg_light": ("FLOAT", {"default": 10, "min": 1, "max": 100, "step": 1}),   
                "noise_removal": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("sketch", "grayscale image")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, image, artist, line_strength, shading_effect, line_color, details, smoothness, enable_bg_color_change, bg_color, bg_light, noise_removal):
        imgNo =  image.shape[0]
        if imgNo == 1:
            print("1 image received for conversion to sketch")
        else:
            print(f"{imgNo} images received for conversion to sketch")
        images = []
        bandwImages = []
        for img in image:
            sketch, bandwI = processImg2Sketch(img, artist, line_strength, shading_effect, line_color, details, smoothness, enable_bg_color_change, bg_color, bg_light, noise_removal)
            images.append(sketch)
            bandwImages.append(bandwI)
        
        images = torch.cat(images, dim=0)
        bandwImages = torch.cat(bandwImages, dim=0)
        print("Conversion complete!")

        return (images, bandwImages)


class LineArt_Assistant():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_thickness": ("INT", {"default": 11, "min": 3, "max": 81, "step": 2}),
                "Clean_up": ("INT", {"default": 7, "min": 1, "max": 49, "step": 2}),
                "deep_Clean_up": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "line_color": (["black", "gray", "white", "lime", "blue", "yellow", "cyan", "magenta", "silver", "red", "maroon", "olive", "green", "purple", "teal", "navy"],),
                "color_strength": ("FLOAT", {"default": 10, "min": 1, "max": 10, "step": 0.1}),
                "details": ("INT", {"default": 9, "min": 1, "max": 9, "step": 1}),
                "smoothness": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1}),  
                "bg_color": (["white", "black", "red", "lime", "blue", "yellow", "cyan", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"],),  
                "bg_light": ("FLOAT", {"default": 10, "min": 1, "max": 100, "step": 1}), 
            },
            #  "optional": {              
            #     "from_LineArt_Preprocessor": ("IMAGE",),
            # },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("lineArt", "grayscale image")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, image, line_thickness, Clean_up, deep_Clean_up, line_color, color_strength, details, smoothness, bg_color, bg_light):
        images = []
        bandwImages = []

        # if use_preprocessors:
        #     imgNo =  from_LineArt_Preprocessor.shape[0] 
        #     if imgNo == 1:
        #         print("1 image received from preprocessor")
        #     else:
        #         print(f"{imgNo} images received from preprocessor")
        #     for img in from_LineArt_Preprocessor:
        #         sketch, bandwI = processLineArt(img, 240, deep_Clean_up, line_color, color_strength, bg_color, bg_light)
        #         images.append(sketch)
        #         bandwImages.append(bandwI)

        # else:
        imgNo =  image.shape[0]
        if imgNo == 1:
            print("1 image received for conversion to lineart")
        else:
            print(f"{imgNo} images received for conversion to lineart")
            
        for img in image:
            sketch, bandwI = processImg2LineArt(img, line_thickness, Clean_up, deep_Clean_up, line_color, color_strength, details, smoothness, bg_color, bg_light)
            images.append(sketch)
            bandwImages.append(bandwI)
        

        images = torch.cat(images, dim=0)
        bandwImages = torch.cat(bandwImages, dim=0)
        print("Conversion complete!")

        return (images, bandwImages)
    

class LineArt_Assistant_2():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lineArt": ("IMAGE",),
                "details": ("INT", {"default": 240, "min": 25, "max": 255, "step": 1}),
                "clean_up": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "line_color": (["black", "gray", "white", "lime", "blue", "yellow", "cyan", "magenta", "silver", "red", "maroon", "olive", "green", "purple", "teal", "navy"],),
                "color_strength": ("FLOAT", {"default": 8, "min": 1, "max": 10, "step": 0.1}),               
                "bg_color": (["white", "black", "red", "lime", "blue", "yellow", "cyan", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"],),  
                "bg_light": ("FLOAT", {"default": 10, "min": 1, "max": 100, "step": 1}), 
                "invert": ("BOOLEAN", { "default": True }),
                "invert_default": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("lineArt", "default")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, lineArt, clean_up, line_color, color_strength, details, bg_color, bg_light, invert, invert_default):
        images = []
        bandwImages = []
  
        imgNo =  lineArt.shape[0] 
        if imgNo == 1:
            print("1 image received")
        else:
            print(f"{imgNo} images")
        for img in lineArt:
            sketch, bandwI = processLineArt(img, details, clean_up, line_color, color_strength, bg_color, bg_light, invert, invert_default)
            images.append(sketch)
            bandwImages.append(bandwI)

        images = torch.cat(images, dim=0)
        bandwImages = torch.cat(bandwImages, dim=0)

        return (images, bandwImages)



def processImg2SketchGrayScale(image, line_strength, shading_effect, details, smoothness, contrast, noise_removal):
    if image is None:
        raise ValueError("Input image is required")

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()
        image = convert_to_uint8(image)

    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D numpy array")
    
    img_height, img_width = image.shape[:2]
    # print(f"The width of the image is: {img_width}, and the height is: {img_height}")


    image = cv2.resize(image, (int(img_width * smoothness), int(img_height * smoothness)), interpolation=cv2.INTER_AREA)

    img_height2, img_width2 = image.shape[:2]

    line_strength = 11 - line_strength

    line_intensity = line_strength / 10

    contrast = contrast/100

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurEffect_s = shading_effect
    blurred_image_s = cv2.GaussianBlur(gray_image, (blurEffect_s, blurEffect_s), 0)
    sketch = cv2.divide(gray_image, blurred_image_s, scale=256.0)
    sketch = cv2.resize(sketch, (int(img_width2 * (1 / smoothness)), int(img_height2 * (1 / smoothness))), interpolation=cv2.INTER_AREA)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    sketch = darken_image(sketch, details, line_intensity)
    sketch = contrast_image(sketch, details, contrast)
    if noise_removal > 0:
        sketch = remove_noise_from_sketch(sketch, noise_removal)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    

    output_image_tensor = torch.from_numpy(sketch).permute(2, 0, 1).unsqueeze(0).float()/255
    output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)


    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    gray_image = torch.from_numpy(gray_image).permute(2, 0, 1).unsqueeze(0).float()/255
    gray_image = gray_image.permute(0, 2, 3, 1)

    return output_image_tensor, gray_image


def processImg2Sketch(image, artist, line_strength, shading_effect, line_color, details, smoothness, enable_bg_color_change, bg_color, bg_light, noise_removal):
    if image is None:
        raise ValueError("Input image is required")

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()
        image = convert_to_uint8(image)

    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D numpy array")
    
    img_height, img_width = image.shape[:2]
    # print(f"The width of the image is: {img_width}, and the height is: {img_height}")

    line_intensity = 11 - line_strength

    bg_light = bg_light/10

    image = cv2.resize(image, (int(img_width * smoothness), int(img_height * smoothness)), interpolation=cv2.INTER_AREA)

    img_height2, img_width2 = image.shape[:2]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    blurEffect_s = shading_effect
    blurred_image_s = cv2.GaussianBlur(gray_image, (blurEffect_s, blurEffect_s), 0)
    sketch = cv2.divide(gray_image, blurred_image_s, scale=256.0)
    sketch = cv2.resize(sketch, (int(img_width2 * (1 / smoothness)), int(img_height2 * (1 / smoothness))), interpolation=cv2.INTER_AREA)
    if noise_removal > 0:
        sketch = remove_noise_from_sketch(sketch, noise_removal)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    if artist == "1":
        sketch = new_ink(sketch, details, line_intensity, getColor(line_color), True)
    else: 
        sketch = new_ink2(sketch, details, line_intensity, getColor(line_color)) 

    if enable_bg_color_change:
        sketch = new_bg(sketch, details, bg_light, getColor(bg_color))

    output_image_tensor = torch.from_numpy(sketch).permute(2, 0, 1).unsqueeze(0).float()/255
    output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)


    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    gray_image = torch.from_numpy(gray_image).permute(2, 0, 1).unsqueeze(0).float()/255
    gray_image = gray_image.permute(0, 2, 3, 1)

    return output_image_tensor, gray_image


def processImg2LineArt(image, line_thickness, Clean_up, deep_Clean_up, line_color, color_strength, details, smoothness, bg_color, bg_light):
    if image is None:
        raise ValueError("Input image is required")

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()
        image = convert_to_uint8(image)

    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D numpy array")
    
    img_height, img_width = image.shape[:2]
    # print(f"The width of the image is: {img_width}, and the height is: {img_height}")


    image = cv2.resize(image, (int(img_width * smoothness), int(img_height * smoothness)), interpolation=cv2.INTER_AREA)

    img_height2, img_width2 = image.shape[:2]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    line_intensity = 11 - color_strength

    detail = 10-details

    bg_light = bg_light/10

    if Clean_up > 1:
        blurEffect_s = Clean_up
        blurred_image_s = cv2.GaussianBlur(gray_image, (blurEffect_s, blurEffect_s), 0)  
        blurred_image_s = convert_to_uint8(blurred_image_s)
        sketch = cv2.adaptiveThreshold(blurred_image_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, line_thickness, detail)
    else:
        gray_imageN = convert_to_uint8(gray_image)
        sketch = cv2.adaptiveThreshold(gray_imageN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, line_thickness, detail)

    sketch = cv2.resize(sketch, (int(img_width2 * (1 / smoothness)), int(img_height2 * (1 / smoothness))), interpolation=cv2.INTER_AREA)
    if deep_Clean_up > 0:
        sketch = remove_noise_from_sketch(sketch, deep_Clean_up)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    sketch = new_ink2(sketch, details, line_intensity, getColor(line_color)) 
    sketch = new_bg(sketch, details, bg_light, getColor(bg_color))


    output_image_tensor = torch.from_numpy(sketch).permute(2, 0, 1).unsqueeze(0).float()/255
    output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    gray_image = torch.from_numpy(gray_image).permute(2, 0, 1).unsqueeze(0).float()/255
    gray_image = gray_image.permute(0, 2, 3, 1)

    return output_image_tensor, gray_image  


def processLineArt(image, details, deep_Clean_up, line_color, color_strength, bg_color, bg_light, invert = True, invert_default = True):
    if image is None:
        raise ValueError("Input image is required")

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()
        image = convert_to_uint8(image)

    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D numpy array")
    
    originalLineArt = image

    sketch = np.copy(originalLineArt)
    
    if invert:
        sketch = 255 - sketch

    if invert_default:
        originalLineArt = 255 - originalLineArt
    

    line_intensity = 11 - color_strength

    bg_light = bg_light/10

    

    if deep_Clean_up > 0:
        sketch = remove_noise_from_sketch(sketch, deep_Clean_up)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    sketch = new_ink2(sketch, details, line_intensity, getColor(line_color)) 
    sketch = new_bg(sketch, details, bg_light, getColor(bg_color))

    output_image_tensor = torch.from_numpy(sketch).permute(2, 0, 1).unsqueeze(0).float()/255
    output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

    originalLineArt = torch.from_numpy(originalLineArt).permute(2, 0, 1).unsqueeze(0).float()/255
    originalLineArt = originalLineArt.permute(0, 2, 3, 1)

    return output_image_tensor, originalLineArt

    

NODE_CLASS_MAPPINGS = {
    "Sketch_Assistant_grayScale" : Sketch_Assistant_grayScale,
    "Sketch_Assistant" : Sketch_Assistant,
    "LineArt_Assistant": LineArt_Assistant,
    "LineArt_Assistant_2": LineArt_Assistant_2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sketch_Assistant_grayScale" :"Img2Sketch Assistant (Grayscale)",
    "Sketch_Assistant" :"Img2Sketch Assistant ",
    "LineArt_Assistant": "Img2LineArt Assistant",
    "LineArt_Assistant_2": "lineArt2LineArt Assistant",
    
}
