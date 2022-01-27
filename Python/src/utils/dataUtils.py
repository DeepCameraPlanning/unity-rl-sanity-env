import torch

# TODO: a better way to refactor it
def stat_interpret_O3_C3(self, state):
  return [state[:, :3], state[:, 3:]]

def stat_interpret_O3_C3Vc1(self, state):
  return [state[:, :3], state[:, 3:]]