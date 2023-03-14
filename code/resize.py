from PIL import Image

img = Image.open('Image/sample.png')
img = img.resize((1920,1080), Image.Resampling.LANCZOS)
img.save('Image/sample.png')