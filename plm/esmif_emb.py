import esm
import numpy as np
import torch
import os


def struct_emb(pdb_file, feature_path, device: str = 'cpu', t_num: int = 4, redo=True):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device)
    torch.set_num_threads(t_num)

    if not (os.path.isfile(feature_path) and os.path.getsize(feature_path)) or redo:
        print(f"Calculating structure embeddings ...")
        structure = esm.inverse_folding.util.load_structure(pdb_file)  # type: ignore
        batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
        coords, model_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)  # type: ignore
        batch = [(coords, None, None)]
        coords, confidence, _, _, padding_mask = batch_converter(batch)
        encoder_out = model.encoder.forward(coords.to(device), padding_mask.to(device), confidence.to(device),
                                            return_all_hiddens=False)  # type: ignore
        representations = encoder_out['encoder_out'][0][1:-1, 0].detach().cpu().numpy()
        np.save(feature_path, representations)