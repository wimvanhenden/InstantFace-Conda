#https://www.youtube.com/watch?v=Qzj6Gq_NeTw&t=4156s
#https://github.com/RobMulla/twitch-stream-projects/blob/main/060-insightface/insight-face.ipynb


import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from matplotlib import pyplot as plt

#workaround for numpy
np.int = np.int_

app = FaceAnalysis(name='antelopev2', root="./", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

#img = ins_get_image('t1')
img = cv2.imread('./examples/buddy_orginal.jpg')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

print(faces[0]['bbox'])
print(faces[0]['gender'])

#display image
#plt.figure(1)
#plt.imshow(img[:,:,::-1])
#plt.title('source')
#plt.show()

'''
rob_faces = app.get(milo)
assert len(rob_faces) == 1
rob_face = rob_faces[0]
bbox = rob_face['bbox']
bbox = [int(b) for b in bbox]
plt.figure(2)
plt.imshow(milo[bbox[1]:bbox[3],bbox[0]:bbox[2],::-1])
plt.show()'''

#img = ins_get_image('t1')
#faces = app.get(img)
#rimg = app.draw_on(img, faces)
#cv2.imwrite("./t1_output.jpg", rimg)