from AI_model import cataract_predict as cp
from glob import glob
images = glob('./test/*')
for img in images:
    cp.image_test(img)