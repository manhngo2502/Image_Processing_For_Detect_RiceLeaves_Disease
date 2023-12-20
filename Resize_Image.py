
from PIL import Image

im=Image.open('D:\Test_Cut_Frame\Image/frame_0.jpg')
im = im.resize((521,512), resample=Image.NEAREST)
im.save('test.jpg')