import pandas as pd

from os import listdir
from random import sample, seed
from sklearn.metrics import accuracy_score


CAM_IMAGES = "./data/image/0801-500/train"

class CamUtils:
  seed(1010)
  LABELS = sorted(list(set([d.split("-")[0] for d in listdir(CAM_IMAGES)])))
  L2I = {v:i for i,v in enumerate(LABELS)}
  S = sample(list(range(1000, 3200)), k=2100)

  @staticmethod
  def PRIME_SEED(i):
    return [5003, 5009, 5011, 5021, 5023,
            5039, 5051, 5059, 5077, 5081,
            5087, 5099, 5101, 5107, 5113,
            5119, 5147, 5153, 5167, 5171,
            5179, 5189, 5197, 5209, 5227,
            5231, 5233, 5237, 5261, 5273,
            5279, 5281, 5297, 5303, 5309][i]

  @staticmethod
  def get_label(fname):
    return CamUtils.LABELS.index(fname.split("-")[0])

  @staticmethod
  def function(x):
    if "-" in x:
      return CamUtils.LABELS.index(x.split("-")[0])
    else:
      x_int = int(x.split(".")[0])
      i_idx = [x_int % CamUtils.PRIME_SEED(i) == 0 for i in CamUtils.L2I.values()].index(True)
      return i_idx

  @staticmethod
  def s2i(x):
    l = x.split("-")[0]
    i = int(x.split("-")[1].split(".")[0])
    return CamUtils.S[i] * CamUtils.PRIME_SEED(CamUtils.L2I[l])

  @staticmethod
  def classification_accuracy(filenames, predictions):
    true_labels = [CamUtils.function(l) for l in filenames]
    if (isinstance(predictions, pd.core.frame.DataFrame) or isinstance(predictions, pd.core.series.Series)):
      predictions = predictions.values
    return accuracy_score(true_labels, predictions)
