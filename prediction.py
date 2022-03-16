


input_shape = (64, 64)

def read_image(image):
    pil_image = Image.open(BytesIO(image))
    image = pil_image.resize(input_shape)

    return pred








