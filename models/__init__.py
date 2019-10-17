from .faster_rcnn import *


def get_model(modelName):
    return globals()[modelName]()
