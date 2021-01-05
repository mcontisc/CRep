# CRep: Python code
Copyright (c) 2020 [Hadiseh Safdari](https://github.com/hds-safdari), [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani) and [Caterina De Bacco](http://cdebacco.com).

Implements the algorithm described in:

[1] Safdari H., Contisciani M. & De Bacco C. (2020). *A generative model for reciprocity and community detection in networks*, arXiv:2012.08215. 

If you use this code please cite this [article](https://arxiv.org/abs/2012.08215) (_preprint_).  

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Files
- `main.py` : General version of the algorithm. It performing the inference in the given single-layer directed network. It infer latent variables as community memberships to nodes and a reciprocity parameter to the whole network.
- `CRep.py` : Class definition of CRep, the algorithm to perform inference in networks with reciprocity. The latent variables are related to community memberships and reciprocity value. This code is optimized to use sparse matrices.
- `generate_data.py` : Code for generating the benchmark synthetic data with an intrinsic community structure and a given reciprocity value. 
- `generative_model_reciprocity.py` : Class definition of the reciprocity generative model with the member functions required. It builds a directed, possibly weighted, network. It contains functions to generate networks with an intrinsic community structure and a given reciprocity value, with reciprocity-only or without reciprocity. 
- `tools.py` : Contains non-class functions for handling the data.
- `main_cv.py` : Code for performing a k-fold cross-validation procedure, in order to estimate the hyperparameter **K** (number of communities). It runs with a given K and returns a csv file summarizing the results over all folds. The output file contains the value of the pseudo log-likelihood, the regular AUC and the conditional AUC of the link prediction, both in the train and in test sets.
- `cv_functions.py` : Contains functions for performing the k-fold cross-validation procedure.
- `test.py` : Code for testing the main algorithm.
- `test_cv.py` : Code for testing the cross-validation procedure.
- `setting_syn_data.yaml` : Setting to generate synthetic data (input for *generate_data.py*).
- `setting_CRep.yaml` : Setting to run the algorithm CRep (input for *main.py* and *main\_cv.py*).
- `setting_CRepnc.yaml` : Setting to run the algorithm CRep without normalization constraints on the membership parameters (input for *main.py* and *main\_cv.py*.
- `setting_CRep.yaml` : Setting to run the algorithm CRep without considering  the reciprocity effect (input for *main.py* and *main\_cv.py*.
- `analyse_results.ipynb` : Example jupyter notebook to import the output results.

## Usage
To test the program on the given example file, type

```bash
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

## Output
The algorithm returns a compressed file inside the `./data/output` folder. To load and print the out-going membership matrix:

```bash
import numpy as np 
theta = np.load('theta_Crep.npz')
print(theta['u'])
```

_theta_ contains the two NxK membership matrices **u** *('u')* and **v** *('v')*, the 1xKxK (or 1xK if assortative=True) affinity tensor **w** *('w')*, the reciprocity coefficient **$\eta$** *('eta')*, the total number of iterations *('max_it')*, the value of the maximum pseudo log-likelihood *('maxPSL')* and the nodes of the network *('nodes')*.  

For an example `jupyter notebook` importing the data, see *analyse_results.ipynb*.
