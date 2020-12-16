# CRep reciprocity in networks
Python implementation of CRep algorithm described in:

- [1] Safdari H., Contisciani M. & De Bacco C. (2020). *A generative model for reciprocity in networks*, arXiv:2012.08215.  

This is a new probabilistic generative model and efficient algorithm to model reciprocity in directed networks. It assigns latent variables as community memberships to nodes and a reciprocity parameter to the whole network and it formalizes the assumption that a directed interaction is more likely to occur if an individual has already observed an interaction towards her. <br>

If you use this code please cite [1].   

The paper can be found [here](https://arxiv.org/abs/2012.08215) (_preprint_).  

Copyright (c) 2020 [Hadiseh Safdari](https://github.com/hds-safdari), [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani) and [Caterina De Bacco](http://cdebacco.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's included
- `code` : Contains the Python implementation of CRep algorithm, the code for performing the cross-validation procedure and the code for generating benchmark synthetic data with intrinsic community structure and given reciprocity value.
- `data/input` : Contains an example of directed network having an intrinsic community structure and a given reciprocity value, and some example files to initialize the latent variables. They are synthetic data.
- `data/output` : Contains some results to test the code.

## Requirements
The project has been developed using Python 3.7 with the packages contained in *requirements.txt*. We suggest to create a conda environment with
`conda create --name CRep python=3.7.9 --no-default-packages`, activate it with `conda activate CRep`, and install all the dependencies by running (inside `CRep` directory):

`pip install -r requirements.txt`

## Test
You can run tests to reproduce results contained in `data/output` by running (inside `code` directory):  

```bash
python -m unittest test.py   
python -m unittest test_cv.py
```

## Usage
To test the program on the given example file, type:  

```bash
cd code
python main.py
```

It will use the sample network contained in `./data/input`. The adjacency matrix *syn111.dat* represents a directed, weighted network with **N=600** nodes, **K=3** equal-size unmixed communities with an **assortative** structure and reciprocity parameter **eta=0.5**. 

### Parameters
- **-a** : Model configuration to use (CRep, CRepnc, CRep0), *(default='CRep')*.
- **-K** : Number of communities, *(default=3)*.
- **-A** : Input file name of the adjacency matrix, *(default='syn111.dat')*.
- **-f** : Path of the input folder, *(default='../data/input/')*.
- **-e** : Name of the source of the edge, *(default='source')*.
- **-t** : Name of the target of the edge, *(default='target')*.
- **-d** : Flag to force a dense transformation of the adjacency matrix, *(default=False)*.

You can find a list by running (inside `code` directory): 

```bash
python main.py --help
```

## Input format
The network should be stored in a *.dat* file. An example of rows is

`node1 node2 3` <br>
`node1 node3 1`

where the first and second columns are the _source_ and _target_ nodes of the edge, respectively; the third column tells if there is an edge and the weight. In this example the edge node1 --> node2 exists with weight 3, and the edge node1 --> node3 exists with weight 1.

Other configuration settings can be set by modifying the *setting\_\*_.yaml* files: 

- *setting\_syn_data.yaml* : contains the setting to generate synthetic data
- *setting\_CRep.yaml* : contains the setting to run the algorithm CRep
- *setting\_CRepnc.yaml* : contains the setting to run the algorithm CRep without normalization constraints on the membership parameters
- *setting\_CRep0.yaml* : contains the setting to run the algorithm CRep without considering  the reciprocity effect

## Output
The algorithm returns a compressed file inside the `data/output` folder. To load and print the out-going membership matrix:

```bash
import numpy as np 
theta = np.load('theta_Crep.npz')
print(theta['u'])
```

_theta_ contains the two NxK membership matrices **u** *('u')* and **v** *('v')*, the 1xKxK (or 1xK if assortative=True) affinity tensor **w** *('w')*, the reciprocity coefficient **$\eta$** *('eta')*, the total number of iterations *('max_it')*, the value of the maximum pseudo log-likelihood *('maxPSL')* and the nodes of the network *('nodes')*.  

For an example `jupyter notebook` importing the data, see `code/analyse_results.ipynb`.
