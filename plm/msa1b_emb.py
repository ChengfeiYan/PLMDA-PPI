import torch
import numpy as np
import os
import esm
import warnings
import Bio

def msa_emb_work(feature_path,msa_path,device='cpu',t_num: int = 4,redo=False):
	if t_num:
		torch.set_num_threads(t_num)
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

	cpu_model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
	cpu_model = cpu_model.to(device)

	model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
	model = model.to(device)
	batch_converter = alphabet.get_batch_converter()

	if not (os.path.isfile(feature_path) and os.path.getsize(feature_path)) or redo:
		print(f"Calculating msa embeddings ...")
		# Load msa, max msa count is 256
		count = 0
		msa_data = []
		for r in Bio.SeqIO.parse(msa_path, 'fasta'):  # type: ignore
			if count >= 256:
				break
			msa_data.append((r.description, str(r.seq)))
			count += 1

		batch_labels, batch_seqs, batch_tokens = batch_converter(msa_data)
		with torch.no_grad():
			torch.cuda.empty_cache()
			try:
				batch_tokens = batch_tokens.to(device=device, non_blocking=True)
				out = model(batch_tokens, repr_layers=[12], return_contacts=True)
			except:
				warnings.warn("Device is not usable,using cpu instead.")
				batch_tokens = batch_tokens.to(device='cpu', non_blocking=True)
				out = cpu_model(batch_tokens, repr_layers=[12], return_contacts=True)

			representations = out['representations'][12].cpu().numpy()[0, 0, 1:, :]

			np.save(feature_path, representations)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    