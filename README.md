# PLMDA-PPI:
Mechanism-Aware Protein-Protein Interaction Prediction via Contact-Guided Dual Attention on Protein Language Models:
![image](https://github.com/ChengfeiYan/PLMDA-PPI/blob/main/mainfig.jpg)
## Requirements
- #### python3.9
  1. [pytorch](https://pytorch.org/)
  2. [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning)
  2. [Biopython](https://biopython.org/)
  3. [esm](https://github.com/facebookresearch/esm)
  4. [numpy](https://numpy.org/)
  5. [GVP](https://github.com/drorlab/gvp-pytorch)
  6. [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  


## Installation
### 1. Install PLMGraph-Inter
    git clone https://github.com/ChengfeiYan/PLMDA-PPI.git
### 2. Download the trained models
   Download the trained models from  [trained models](https://drive.google.com/file/d/1prd9KKoM_BAJuzeZm4DWkiUQur-1mdCA/view?usp=sharing).

## Usage
    python predict.py sequenceA msaA pdbA sequenceB msaB pdbB result_path model_path device
    1.  sequenceA: fasta file corresponding to target A.
    2.  msaA: a3m file corresponding to target A (multiple sequence alignment).
    3.  pdbA: pdb file corresponding to target A.
    4.  sequenceB: fasta file corresponding to target B.
    5.  msaB: a3m file corresponding to target B (multiple sequence alignment).
    6.  pdbB: pdb file corresponding to target B.
    7.  result_path: [a directory for the output]
    8.  model_path: PLMDA-PPI(PDB) or PLMDA-PPI(Transfer)
    8.  device: cpu, cuda:0, cuda:1, ...
   Where MSA should be derived from the Uniref100 database. If you encounter that some residues in the pdb file are missing, you can use [MODELLER](https://salilab.org/modeller/tutorial/iterative.html) to fill in these missing residues.

## Example
    python predict.py ./example/1Z6O_C.fasta ./example/1Z6O_C.msa.fasta ./example/1Z6O_C.pdb ./example/1Z6O_O.fasta ./example/1Z6O_O_msa.fasta ./example/1Z6O_O.pdb ./example/result PLMDA-PPI(PDB).pt cpu

## Train
The detailed script used to train PLMDA-PPI is in [main_inter.py](https://github.com/ChengfeiYan/PLMDA-PPI/blob/main/model/main_inter.py), which contains all the details of training PLMDA-PPI, including how to choose the best model, how to calculate the loss, etc.

## Reference  
Please cite: Shuchen Deng, Chengfei Yan. Mechanism-Aware Protein-Protein Interaction Prediction via Contact-Guided Dual Attention on Protein Language Models.

