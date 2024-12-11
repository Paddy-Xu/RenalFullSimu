# Full-scale model of renal hemodynamics using a reconstructed vascular tree

<!-- ![](./fllowchart.png)    -->

# Installation

Run the following command for installation of the project

```
git clone https://github.com/Paddy-Xu/RenalFullSimu

cd RenalFullSimu

conda create -n NephronModel python==3.10
conda activate NephronModel             
pip install -r requirements.txt

```
It will install the required packages listed in requirements.txt.

# Run simulation

## Simulation of autoregulation on a full-scale neprhon-vascular network 

### For simulating autoregulation on the full-scale neprhon-vascular network with inlet pressure range from 80 mmHg to 200 mmHg, run

```
python tree_model.py
```

### Input files
* ```final_tree.vtk``` The most crucial input to the simulation pipeline is a full-scale renal arterial tree structure stored in .vtk format.
Each node has its xyz location while each edge has its radius as edge feature. 
The simulation starts by first computing the blood flow and pressure along the vascular tree without autoregulation by applying Kirchhoff's current law 
as described in the paper. Since this step is quite time-consuming, these can also be precomputed and provide another .vtk file, in which 
the tree contains not only the topological info but also hemodynamic feautres (flow and pressure without autoregulation).
  
* ```surface.nii.gz``` A binary mask of the kidney where the full-scale renal arterial tree is reconstructed from. 
This is ONLY required if we want to differentiate three subpopulations of nephrons based on its depth from kidney surface.


### Input hyper-parameters

The hyper-parameters we used are given in the code already, but you may also want to adjust them if necessary

* ```P_in_range``` Range of the inlet arterial pressure to simulate the autoregulation (in mmHg where 100 is the control case value)
* ```num_iter``` maximum number of iterations when solving autoregulation
* ```lr``` learning rate (for updating AA radii after each iteration)
*  ```only_myo``` flag for simulating myogenic mechanism only by blocking TGF
* ```pop``` flag for simulating 3 subpopulations of nephrons
* ```relTol``` relative tolerance of change of renal blood flow after each iteration to assess convergence (stop iteration)
* Vascular tree related parameters (these should never be changed unless you have a new vascular tree data)
  * ```root_loc``` root location of the tree (xyz coordinates)
  * ```vspace``` voxel size


### Expected output
The code will output the renal blood flow under each arterial pressure level given in ```P_in_range```, both with and without autoregulation.


## Simulation of autoregulation on a single nephron model (for comparison study)

For simulating a single nephron model, run

```
python single_model.py
```

This will calculate the pre-AA resistance of a certain nephron (and its connected AA) and simulate autoregulation of a single nephron. 
Note that a full-scale renal arterial tree structure will still have to be defined and loaded to calculate the pre-AA resistance and initial AA flow and pressure of a certain nephron.

### Input files and hyper-parameters
Please refer to the input to the previous section (tree_model.py) for detailed illustration. Here we just list parameters specific to the single nephron model.

* ```cur_node``` index of a certain nephron number. This can be precomputed based on the initial pressure.


## Code scturectures

[tree_model](tree_model.py) and [single_model](single_model.py):  main entry files for running simulation as described above

[parameters](parameters.py) includes all the parameters that are given in the appendix of the paper

[nephron_eqs](nephron_eqs.py) defines all the equations for both glomerular and tubular model as described in the method section 

[nephron_sovler](nephron_sovler.py) defines the step for sovling the nephron model (both glomerular and tubular model). The final output to the next (afferent_arteriole) model is the  ```Cs_md``` (NaCl concentration at macula densa)

[afferent_arteriole](afferent_arteriole.py) defines the afferent arteriole model as described in the method section (for solving AA radius)

[vascular_tree_model](vascular_tree_model.py) defines how the input vascular tree file in ```.vtk``` is read into python and also defines the 
Kirchhoff's law to solve blood flow and pressure though the vascular tree given a new set of afferent arteriole radii


<!---
Finally, radius is of course an edge feature because it is associated with each vessel segment. 
However, to visualize 3D cylindrical tubes in ParaView via interpolations, you will need to generate radius at each node.
To do so, run save_node_radius.py 
-->
