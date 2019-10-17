import matplotlib.pyplot as plt
import cv2

colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
          (125, 0, 0),
          (0, 125, 0),
          (0, 0, 125), ]


def showImage(image):
    fig = plt.figure(figsize=(20, 30), dpi=50)
    plt.imshow(image)
    plt.figure
    plt.axis('off')
    plt.show()


def showImages(images=[], names=[]):
    fig = plt.figure(figsize=(40, 30), dpi=50)
    for i, image in enumerate(images):
        ax = plt.subplot(1, len(images), i + 1)
        ax.set_title(names[i],fontsize=120)
        ax.axis('off')
        plt.imshow(image)
    plt.show()



def draw_one_image(img, boxes, labels):
    if boxes.shape[0] == labels.shape[0]:
        for i in range(len(labels)):
            box = boxes[i, :]
            label = labels[i]
            color = colors[label]

            hight = box[3] - box[1]

            font_size = 0.8
            thickness = 2

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
            cv2.putText(img, "cl:" + str(label), (box[0], int(box[1] + 0.5 * hight)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, color, thickness)
        return img
    else:
        print('the length of boxes and labels not equal!')
