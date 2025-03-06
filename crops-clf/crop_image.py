from PIL import Image, ImageDraw

# Open an image
image = Image.open('/data/frames/004-squacco_heron/frame_00000.jpg')

# Define the coordinates of the bounding box (left, upper, right, lower)
bounding_box = (490.64, 168.22, 719.26, 694.64)

# Create a drawing context
image = image.crop(bounding_box)

# Optionally, save the image with the bounding box
image.save('image_with_bounding_box.jpg')
