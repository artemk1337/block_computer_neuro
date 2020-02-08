import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import time
from scipy.linalg import norm
import pandas as pd
from PIL import Image

import os
import subprocess


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


cap = cv2.VideoCapture(0)


target = Image.open('target.jpg')
x_aligned1, prob = mtcnn(target, save_path=None, return_prob=True)
embeddings1 = resnet(x_aligned1[:1]).detach().cpu().numpy()

start = time.time()

def neuro(x):
    aligned = []
    # x.save('target.jpg')
    x_aligned, prob = mtcnn(x, save_path=None, return_prob=True)
    if len(prob) == 1 and prob[0] is not None:
        # Уверенность 99% и выше
        if prob[0] > 0.99:
            aligned.append(x_aligned[0])
    elif x_aligned is not None:
        for k in range(len(prob)):
            # Уверенность 99% и выше
            if prob[k] >= 0.99:
                aligned.append(x_aligned[k])
    return aligned


def check_person(aligned, arr):
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu().numpy()
    for i1 in embeddings1:
        arr = [True if norm(i1 - i) < 1 else False for i in embeddings]
        print(arr)
    return arr


while(True):
    arr = []
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame',gray)
    cv2.imshow('frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        aligned = neuro(Image.fromarray(frame))
        arr = check_person(aligned, arr)
    except Exception as e:
        print(e)
        pass
    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for i in range(len(arr)):
        if arr[i] is True:
            start = time.time()
    if (time.time() - start > 15):
        os.system('pmset sleepnow')
    print(time.time() - start)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


