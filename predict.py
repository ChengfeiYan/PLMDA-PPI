import torch
import os
import lightning.pytorch as pl
from model.main_inter import MainModel
from lightning.pytorch.callbacks import BasePredictionWriter
import sys
import numpy as np
import pickle
import pdb_graph
from plm.msa1b_emb import msa_emb_work
from plm.esm1b_emb import seq_emb
from plm.esmif_emb import struct_emb
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_mono_feature(fasA, a3mA, pdbA,fasB, a3mB, pdbB,result_path, device):

    LoadHHM = "./LoadHHM.py"

    # cal graph
    graphA_path = os.path.join(result_path, 'graphA.pkl')
    graphB_path = os.path.join(result_path, 'graphB.pkl')
    pdb_graph.main(pdbA, graphA_path)
    pdb_graph.main(pdbB, graphB_path)

    # reformat msa
    refine_a3mA = os.path.join(result_path, 'A.msa.refine.a3m')
    filter_a3mA = os.path.join(result_path, 'A.msa.filter.fasta')
    os.system(f'hhfilter -i {a3mA} -o {refine_a3mA} -cov 50 -id 100')
    os.system(f'hhfilter -i \'{refine_a3mA}\' -o \'{filter_a3mA}\' -diff 256')

    refine_a3mB = os.path.join(result_path, 'B.msa.refine.a3m')
    filter_a3mB = os.path.join(result_path, 'B.msa.filter.fasta')
    os.system(f'hhfilter -i {a3mB} -o {refine_a3mB} -cov 50 -id 100')
    os.system(f'hhfilter -i \'{refine_a3mB}\' -o \'{filter_a3mB}\' -diff 256')

    # cal msa-1b
    msa1b_a = os.path.join(result_path, 'A.msa.emb.npy')
    msa1b_b = os.path.join(result_path, 'B.msa.emb.npy')
    msa_emb_work(msa1b_a, filter_a3mA, device)
    msa_emb_work(msa1b_b, filter_a3mB, device)

    # cal pssm
    hhmA = os.path.join(result_path, 'A.hhm')
    pssmA = os.path.join(result_path, 'A.pssm.pkl')
    hhmB = os.path.join(result_path, 'B.hhm')
    pssmB = os.path.join(result_path, 'B.pssm.pkl')

    os.system(f'hhmake -i {refine_a3mA} -o {hhmA}')
    os.system(f'python {LoadHHM} {hhmA} {pssmA}')
    os.system(f'hhmake -i {refine_a3mB} -o {hhmB}')
    os.system(f'python {LoadHHM} {hhmB} {pssmB}')

    # cal esm-1b
    reprA_esm1b = os.path.join(result_path, 'A.seq.emb.npy')
    reprB_esm1b = os.path.join(result_path, 'B.seq.emb.npy')
    seq_emb(fasA, reprA_esm1b, device)
    seq_emb(fasB, reprB_esm1b, device)

    # cal esm-if1
    reprA_esmif = os.path.join(result_path, 'A.str.npy')
    reprB_esmif = os.path.join(result_path, 'B.str.npy')
    struct_emb(pdbA, reprA_esmif, device)
    struct_emb(pdbB, reprB_esmif, device)

    # load feature

    features = []
    seq_embA = np.load(reprA_esm1b)
    seq_embA = torch.from_numpy(seq_embA)

    graph = pickle.load(open(graphA_path, 'rb'))

    msa_emb = np.load(msa1b_a)
    features.append(torch.from_numpy(msa_emb))

    pssm = pickle.load(open(pssmA, 'rb'))['PSSM']
    features.append(torch.from_numpy(pssm))

    structure_emb = np.load(reprA_esmif)
    features.append(torch.from_numpy(structure_emb))

    nodes1 = (
        torch.hstack((graph["nodes_sact"], *features)),
        graph["nodes_vec"]
    )
    edges1 = (graph["edge_scat"], graph["edge_vec"])
    edge_index1 = graph["edge_index"]

    features = []
    seq_embB = np.load(reprB_esm1b)
    seq_embB = torch.from_numpy(seq_embB)

    graph = pickle.load(open(graphB_path, 'rb'))

    msa_emb = np.load(msa1b_b)
    features.append(torch.from_numpy(msa_emb))

    pssm = pickle.load(open(pssmB, 'rb'))['PSSM']
    features.append(torch.from_numpy(pssm))

    structure_emb = np.load(reprB_esmif)
    features.append(torch.from_numpy(structure_emb))

    nodes2 = (
        torch.hstack((graph["nodes_sact"], *features)),
        graph["nodes_vec"]
    )
    edges2 = (graph["edge_scat"], graph["edge_vec"])
    edge_index2 = graph["edge_index"]

    return (nodes1, edges1, edge_index1, nodes2, edges2, edge_index2, seq_embA, seq_embB)


class PPIDataset(Dataset):
    def __init__(self, fasA, a3mA, pdbA,fasB, a3mB, pdbB,result_path, device) -> None:
        super().__init__()
        self.fasA = fasA
        self.a3ma = a3mA
        self.pdbA = pdbA
        self.fasB = fasB
        self.a3mb = a3mB
        self.pdbB = pdbB
        self.result_path = result_path
        self.device = device

    def __getitem__(self, index):
        nodes1, edges1, edge_index1, nodes2, edges2, edge_index2, seq1, seq2 = get_mono_feature(fasA, a3mA, pdbA, fasB, a3mB, pdbB, result_path, device)
        return (nodes1, edges1, edge_index1, nodes2, edges2, edge_index2, seq1, seq2)

    def __len__(self):
        return 1
class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir


    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        contact_map, pred_interaction = prediction
        contact_map = contact_map.cpu()
        contact_map = contact_map.data.numpy()
        pred_interaction = pred_interaction.cpu()
        pred_interaction = pred_interaction.data.numpy()
        np.savetxt(os.path.join(self.output_dir, f"pred_interaction.txt"),pred_interaction)
        np.savetxt(os.path.join(self.output_dir, f"pred_contact.txt"), contact_map.squeeze())


if __name__ == "__main__":
    fasA, a3mA, pdbA, fasB, a3mB, pdbB, result_path, model_ckpt, device = sys.argv[1:]
    device = torch.device(device)
    torch.set_float32_matmul_precision('medium')

    hyper_params = {
        "ratios": [100, 1],  # ratios for [map,inter]
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "epochs": 20,
        "model": {
            "graph_module": {
                "num_gvp_layers": 3
            },
            "contact_module": {
                "block_num": 12,
                "input_channels": 896,
                "hidden_channels": 96
            },
            "interaction_module": {
                "input_dim": 1280,
                "hidden_dim": 448,
                "dma_depth": 2
            }
        },
        "dataset": {
            "num_workers": 32,
            "settings": {
                "max_length": 0,
                "true_only": False,
                "features": ["graph", "seq_emb_2b", "msa_emb", "pssm", "structure_emb"]
            }
        },
        "devices": device,
        "precision": 32,
        "load_ckpt": model_ckpt,
    }

    model = MainModel(hyper_params=hyper_params)
    dataset = PPIDataset(fasA, a3mA, pdbA,fasB, a3mB, pdbB,result_path, device)
    dataloader = DataLoader(dataset, batch_size=1)
    pred_writer = CustomWriter(output_dir=result_path, write_interval="batch")

    trainer = pl.Trainer(
        callbacks=[pred_writer],
        # devices=hyper_params["devices"],
        accelerator = "cpu",
        precision=hyper_params["precision"],
        log_every_n_steps=1,
    )

    trainer.predict(model=model, dataloaders=dataloader, ckpt_path=hyper_params["load_ckpt"])
