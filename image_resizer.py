import cv2

def image_preprocess_resizing(image, square_size=512):
    height, width, channels = image.shape
    padding = (0, 0, 0, 0)
    if height == square_size and width == square_size:
        return image, (0, 0, 0, 0)

    if height > width:
        padding = height - width
        image = cv2.copyMakeBorder(image, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padding = (0, 0, 0, padding)

    elif width > height:
        padding = width - height
        image = cv2.copyMakeBorder(image, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padding = (0, 0, padding, 0)
        
    image = cv2.resize(image, (square_size, square_size))
    padding = [padding[0] * square_size / width, padding[1] * square_size / width, padding[2] * square_size / width, padding[3] * square_size / width]
    return image, padding, (height, width)


def image_postprocess_resizing(image, padding, original_size, upscale_factor):
    original_size = (original_size[1] * upscale_factor, original_size[0] * upscale_factor)
    
    if padding != (0, 0, 0, 0):
        padding = (int(padding[0]*upscale_factor), int(padding[1]*upscale_factor), int(padding[2]*upscale_factor), int(padding[3]*upscale_factor))
        image = image[padding[0]:image.shape[0] - padding[2], padding[1]:image.shape[1] - padding[3]]

    image = cv2.resize(image, original_size)
    return image