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
  7. [hh-suite](https://github.com/soedinglab/hh-suite)
  


## Installation
### 1. Install PLMGraph-Inter
    git clone https://github.com/ChengfeiYan/PLMDA-PPI.git
### 2. Download the trained models
   Download the trained models from  [trained models](https://drive.google.com/file/d/1prd9KKoM_BAJuzeZm4DWkiUQur-1mdCA/view?usp=sharing).

## Usage
For pair inference:

    python predict.py sequenceA msaA pdbA sequenceB msaB pdbB result_path model_path device
    1.  sequenceA: fasta file corresponding to target A.
    2.  msaA: a3m file corresponding to target A (multiple sequence alignment).
    3.  pdbA: pdb file corresponding to target A.
    4.  sequenceB: fasta file corresponding to target B.
    5.  msaB: a3m file corresponding to target B (multiple sequence alignment).
    6.  pdbB: pdb file corresponding to target B.
    7.  result_path: [a directory for the output]
    8.  model_path: PLMDA-PPI(PDB) or PLMDA-PPI(Transfer)
    9.  device: cpu, cuda:0, cuda:1, ...
   If you encounter that some residues in the pdb file are missing, you can use [MODELLER](https://salilab.org/modeller/tutorial/iterative.html) to fill in these missing residues.

#### Example
    python predict.py 1Z6O_C.fasta 1Z6O_C.msa.a3m 1Z6O_C.pdb 1Z6O_O.fasta 1Z6O_O_msa.a3m 1Z6O_O.pdb result PLMDA-PPI(PDB).pt cpu

For batch-run inference:

    python predict_list.py ppi_list.csv result_path model_path device
    Where ppi_list.csv is a csv file of:
    {protein_pair},{fasA},{a3mA},{pdbA},{fasB},{a3mB},{pdbB}
    e.g.
    1Z6O_C:1Z6O_O,1Z6O_C.fasta,1Z6O_C.msa.a3m,1Z6O_C.pdb,1Z6O_O.fasta,1Z6O_O_msa.a3m,1Z6O_O.pdb
The example test [csv file](https://github.com/ChengfeiYan/PLMDA-PPI/blob/main/example/example_test.csv) is listed in the example directory.

## Train
The detailed script used to train PLMDA-PPI is in [main_inter.py](https://github.com/ChengfeiYan/PLMDA-PPI/blob/main/model/main_inter.py), which contains all the details of training PLMDA-PPI, including how to choose the best model, how to calculate the loss, etc.

For batch-run inference:

    python train.py ppi_list.csv result_path device
    Where ppi_list.csv is a csv file of:
    {protein_pair},{len1},{len2},{fasA},{a3mA},{pdbA},{fasB},{a3mB},{pdbB},{interaction},{contact}
    1. contact: txt file of true protein pair contact map.
    e.g.
    1Z6O_C:1Z6O_O,212,191,1Z6O_C.fasta,1Z6O_C.msa.a3m,1Z6O_C.pdb,1Z6O_O.fasta,1Z6O_O_msa.a3m,1Z6O_O.pdb,1,1Z6O_C_O.contact
The example train [csv file](https://github.com/ChengfeiYan/PLMDA-PPI/blob/main/example/example_train.csv) is listed in the example directory.

## Data

The `data/` directory contains the PPI datasets and prediction scores used for model training, independent evaluation, and result reproduction. Each record represents a protein pair. In the `interaction` or `Interaction` column, `1` denotes a known interacting pair (positive sample) and `0` denotes a constructed non-interacting pair (negative sample). Training and evaluation datasets use an approximately **10:1** negative-to-positive ratio to reflect PPI sparsity. Candidate negative pairs sharing a known interaction partner were excluded.

### Sources and dataset construction

- **Structure-informed PDB data:** Direct PPIs were extracted from heteromeric PDB complexes deposited before 2022-10-24 for initial training and validation. Retained entries were determined by X-ray diffraction, had a resolution of at most 4 Å, and contained at least two protein chains. An inter-protein residue contact was defined by a heavy-atom distance of at most 8 Å. After filtering by protein length, missing residues, self-interactions, and redundancy, the dataset contained 6,189 non-redundant positive PPIs with residue-contact supervision.
- **Literature-curated HINT data:** High-confidence binary PPIs from HINT were used for human fine-tuning and cross-species evaluation. The datasets cover *H. sapiens*, *M. musculus*, *A. thaliana*, *D. melanogaster*, and *S. pombe*. Self-interactions, proteins outside the 30--1,024 aa range, and interactions involving protein isoforms were removed.
- **Leakage-aware splitting:** Protein sequences were clustered with CD-HIT at a 40% sequence-identity threshold before cluster-level training, validation, and test splits were created. Pairs spanning two groups were excluded; consequently, validation/test proteins share no more than 40% sequence identity with training proteins.
- **Structural inputs:** PDB samples use experimentally resolved monomer structures. For HINT proteins without an experimental structure, AlphaFoldDB-predicted monomer models were used to construct input features.

### Directory and file guide

| Path | Contents | Primary use |
| --- | --- | --- |
| `data/PDB_model/PDB_train.csv` | PDB training protein pairs, including positive and negative samples | Initial training |
| `data/PDB_model/PDB_val.csv` | PDB validation protein pairs, including positive and negative samples | Model selection and PDB validation |
| `data/PDB_model/PDB_newly_deposited.csv` | Independent PDB protein pairs deposited after 2022-10-24 | Time-split generalization evaluation |
| `data/HINT_model/H. sapiens_train.csv` | Human HINT training protein pairs, including positive and negative samples | Transfer-learning fine-tuning |
| `data/HINT_model/H. sapiens_val.csv` | Human HINT validation protein pairs, including positive and negative samples | Fine-tuning model selection |
| `data/HINT_model/H. sapiens_test.csv` | Human HINT test protein pairs, including positive and negative samples | Independent human test set |
| `data/PDB_model/HINT_result/` | Predictions from the PDB-pretrained model on the five HINT species datasets | Cross-species generalization results |
| `data/HINT_model/HINT_result/` | Predictions from the HINT human-fine-tuned model on the five species datasets | Transfer-learning result reproduction |
| `data/*_model/PDB_result/` | Results from both model variants on the PDB validation set and newly deposited PDB test set | PDB benchmark and time-split evaluation |
| `data/sampled_test/` | Per-species subsets and scores from structure-prediction baselines | Comparison with AF2Complex, RF2-Lite, and RF2-PPI |

Training and test index files generally use the columns `protein1,protein2,interaction`. Result files use `pair` as the first column, followed by prediction scores from the evaluated methods. Under `HINT_result`, `full`, `cluster_40`, `cluster_30`, and `cluster_20` denote the full test set and increasingly stringent homology-reduced sets retaining proteins with no more than 40%, 30%, and 20% sequence identity to training proteins, respectively.

## Reference  
Please cite:  Mechanism-Aware Inductive Bias Enhances Generalization in Protein-Protein Interaction Prediction
Shuchen Deng, Xuanjun Wan, Zichun Mu, Sheng-You Huang*, Chengfei Yan*
bioRxiv 2025.07.04.663157; doi: https://doi.org/10.1101/2025.07.04.663157
