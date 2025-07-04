import torch
import numpy as np
import os
import esm
from typing import Tuple
from Bio import SeqIO

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.id, str(record.seq)

    
def seq_emb(fasta_file, feature_path, device: str = 'cpu',t_num: int = 32,  redo=False):

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    if t_num:
        torch.set_num_threads(t_num)
    model = model.to(device)
    seq_data = [read_sequence(fasta_file)]  # type: ignore
    if not (os.path.isfile(feature_path) and os.path.getsize(feature_path)) or redo:
        print(f"Calculating sequence embeddings ...")
        batch_labels, batch_seqs, batch_tokens = batch_converter(seq_data)
        batch_lens = [len(s) for s in batch_seqs]
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device=device, non_blocking=True)
            out = model(batch_tokens, repr_layers=[33], return_contacts=True)
            representations = out['representations'][33].cpu().numpy()[0, 1:batch_lens[0] + 1, :]
            np.save(feature_path, representations)