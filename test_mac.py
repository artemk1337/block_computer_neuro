from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np


import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()



quit()



mtcnn = MTCNN(image_size=160, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


aligned = []
ids = []
link = []


x_aligned, prob = mtcnn("<photo>", save_path=None, return_prob=True)
if len(prob) == 1 and prob[0] is not None:
    # Уверенность 99% и выше
    if prob[0] > 0.99:
        aligned.append(x_aligned[0])
        ids.append(id)
        link.append(current_link)
elif x_aligned is not None:
    for k in range(len(prob)):
        # Уверенность 99% и выше
        if prob[k] >= 0.99:
            aligned.append(x_aligned[k])
            ids.append(id)
            link.append(current_link)

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu().tolist()  # numpy()
dists = [[norm(e1 - e2) for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=ids, index=ids))
