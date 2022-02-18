from typing import Sequence

from omegaconf import DictConfig, OmegaConf
import rich.tree
import rich.syntax
import numpy as np
import plotly.graph_objects as go
import plotly as py
import torch
from scipy import ndimage

fig = go.Figure()
counterViz=0
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        # "compnode",
        "model",
        "env",
        "xp_name",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """
    Adapted from: https://github.com/ashleve/lightning-hydra-template.
    Prints content of DictConfig using Rich library and its tree structure.
    :param config: configuration composed by Hydra.
    :param fields: determines which main fields from config will be printed and
        in what order.
    :param resolve: whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)

# for testing
def testViz():
    dims=np.array([10,10,10])
    sampl = np.random.normal(0.5, 0.1, size=dims) 
    # make sparse
    sampl = np.where(sampl>0.7, sampl, 0)
    sampl1d = reshape_occupancy_grid_inverse(sampl)
    viz_occupancy_grid3D(sampl1d,dims)

def reshape_occupancy_grid_inverse(ocp_grid_3d: np.array):
    return ocp_grid_3d.flatten()
    
def reshape_occupancy_grid(ocp_grid_1d: np.array, 
                           dims:        np.array):
    # asserting a 1 dim 
    # assert(np.shape(ocp_grid_1d)[0]==1)
    # print("shape: ", np.shape(ocp_grid_1d))
    locp_grid_3d = []
    for row in ocp_grid_1d:
      # print("row ", np.shape(row.flatten()))
      row3d = row.flatten().reshape(tuple(dims))
      # print("row3d ", np.shape(row3d))
      locp_grid_3d.append(np.array([row3d]))
    
    return torch.Tensor(np.array(locp_grid_3d))

# TODO!!!
def reshape_occupancy_grid_tensor(ocp_grid_1d: torch.Tensor, 
                                  dims:        torch.Tensor):
  # asserting a 1 dim 
  # assert(np.shape(ocp_grid_1d)[0]==1)
  # print("shape: ", np.shape(ocp_grid_1d))
  locp_grid_3d = []
  for row in ocp_grid_1d:
    # print("row ", np.shape(row.flatten()))
    row3d = row.flatten().reshape(tuple(dims))
    # print("row3d ", np.shape(row3d))
    locp_grid_3d.append(row3d)
  
  return np.array(locp_grid_3d)
    
def viz_occupancy_grid3D(ocp_grid_1d: np.array, 
                         dims:     np.array):
    global counterViz
    global fig
    if(counterViz<1000):
      counterViz+=1
      return
    ocp_grid_3d = reshape_occupancy_grid(ocp_grid_1d, dims)
    lx, ly, lz = dims[0], dims[1], dims[2]
    X, Y, Z = np.mgrid[0:lx, 0:ly, 0:lz]
    fig.data = []
    
    fig.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=ocp_grid_1d.flatten(),
    isomin=0.1,
    isomax=1,
    opacity=0.2, # needs to be small to see through all surfaces
    # caps= dict(x_show=False, y_show=False, z_show=False), # no caps
    surface_count=21, # needs to be a large number for good volume rendering
    colorscale='Greys',
    ))
    
    # fig = go.Figure(data=[go.Scatter3d(
    # x=X.flatten(),
    # y=Y.flatten(),
    # z=Z.flatten(),
    # mode='markers',
    # marker=dict(
    #     size=10,
    #     color=ocp_grid_1d.flatten(),                # set color to an array/list of desired values
    #     colorscale='Greys',   # choose a colorscale
    #     opacity=0.3,
    #     line=dict(
    #     color='red',
    #       width=2
    #       )
    # ),
    # marker_symbol='square'

    # )])


    fig.write_html("/Users/triocrossing/INRIA/UnityProjects/DQN_PL/unity-rl-sanity-env/Python/outputs/VOGtest"+'{:03d}'.format(counterViz)+".html")
    counterViz+=1
    # fig.show()

# computing distance 
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  

def center_of_array_1d(A, dims):
  global counterViz
  if(counterViz!=1000):
      counterViz+=1
      return
  ocp_grid_3d = reshape_occupancy_grid(A, dims)
  mass = ndimage.center_of_mass(ocp_grid_3d[0,0].numpy()) 
  npmass = np.array([mass])
  # print("npmass: ", npmass)
  with open("/Users/triocrossing/INRIA/UnityProjects/DQN_PL/unity-rl-sanity-env/Python/outputs/outCubeCM.csv", "a") as f:
    np.savetxt(f, npmass, delimiter=",")
    # f.write(b"\n")
  return 