import PIL.Image

def imshow(pixels):
    PIL.Image.fromarray(pixels).show()