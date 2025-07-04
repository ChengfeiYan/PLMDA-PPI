from torch import Tensor

import torch
import torch.nn as nn

class InteractionModule(nn.Module):
	def __init__(self,input_dim:int,hidden_dim:int,dma_depth:int) -> None:
		super().__init__()
		self.hidden_dim = hidden_dim

		self.input = nn.Sequential(
				nn.Linear(input_dim,hidden_dim),
				nn.LeakyReLU(negative_slope=0.1, inplace=False),
				nn.LayerNorm(normalized_shape=[hidden_dim])
				)
		
		self.dma_depth = dma_depth

		self.drop = nn.Dropout(p=0.6, inplace=False)

		self.dma = nn.ModuleList([DMA(hidden_dim) for depth in range(dma_depth)])

		self.gru = nn.GRUCell(hidden_dim,hidden_dim)

		self.output = nn.Sequential(
			nn.LayerNorm(normalized_shape=[hidden_dim]),
			nn.Linear(hidden_dim,1),
			nn.Sigmoid())


	def forward(self,protein1:Tensor,protein2:Tensor,contact_map:Tensor)->Tensor:
		'''
		protein1:(N,L1,448)
		protein2:(N,L2,448)
		contact_map:(N,L1,L2)
		return:(N)
		'''
		protein1 = self.input(protein1)
		protein2 = self.input(protein2)

		protein1 = self.drop(protein1)
		protein2 = self.drop(protein2)

		p1 = protein1.mean(dim=1)#(N,hidden_dim)
		p2 = protein2.mean(dim=1)#(N,hidden_dim)

		memory = torch.layer_norm(p1*p2,normalized_shape=[self.hidden_dim])  # (N,hidden_dim)

		for depth in range(self.dma_depth):
			
			f1,f2 = self.dma[depth](protein1,protein2,contact_map,memory)#(N,hidden_dim)

			memory = torch.layer_norm(self.gru(memory,f1*f2),normalized_shape=[self.hidden_dim])

		interaction = self.output(torch.layer_norm(f1*f2,normalized_shape=[self.hidden_dim])).squeeze(1)  # (N)#type:ignore

		return interaction
	

class DMA(nn.Module):
	def __init__(self,hidden_dim) -> None:
		super().__init__()
		self.hidden_dim = hidden_dim
		self.transform = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(normalized_shape=[hidden_dim]),
			nn.LeakyReLU(negative_slope=0.01, inplace=False),
			)
		self.m1 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(normalized_shape=[hidden_dim]),
			nn.LeakyReLU(negative_slope=0.01, inplace=False),
		)
		self.h0 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(normalized_shape=[hidden_dim]),
			nn.LeakyReLU(negative_slope=0.01, inplace=False),
		)
		self.h1 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.Sigmoid()
		)

	def forward(self,protein1:Tensor,protein2:Tensor,contact_map:Tensor,memory:Tensor)->tuple[Tensor,Tensor]:
		'''
		protein1:(N,L1,hidden_dim)
		protein2:(N,L2,hidden_dim)
		contact_map:(N,L1,L2)
		memory:(N,hidden_dim)
		return:(N,hidden_dim),(N,hidden_dim)
		'''
		#protein1=protein1.detach()
		#protein2=protein2.detach()
		#contact_map=contact_map.detach()#stop backprop

		_1to2 = torch.layer_norm(torch.matmul(contact_map.transpose(1,2),self.transform(protein1)), [self.hidden_dim])#(N,L1,L2)->(N,L2,L1),(N,L2,L1)*(N,L1,hidden_dim)->(N,L2,hidden_dim)
		_2to1 = torch.layer_norm(torch.matmul(contact_map,self.transform(protein2)), [self.hidden_dim])#(N,L1,L2)*(N,L2,hidden_dim)->(N,L1,hidden_dim)

		tmp1 = torch.layer_norm((self.h0(protein1)*self.m1(memory).unsqueeze(dim=1))*_2to1, [self.hidden_dim])#(N,L1,hidden_dim)*(N,1,hidden_dim)*(N,L1,hidden_dim)=(N,L1,hidden_dim)
		tmp2 = torch.layer_norm((self.h0(protein2)*self.m1(memory).unsqueeze(dim=1))*_1to2, [self.hidden_dim])  # (N,L2,hidden_dim)*(N,1,hidden_dim)*(N,L2,hidden_dim)=(N,L2,hidden_dim)

		attn1 = self.h1(tmp1)#(N,L1,hidden_dim)
		attn2 = self.h1(tmp2)#(N,L2,hidden_dim)

		#(N,L1,hidden_dim)*(N,L1,hidden_dim)=(N,L1,hidden_dim)->(N,hidden_dim)
		f1 = torch.layer_norm(torch.mean(protein1*attn1,dim=1), [self.hidden_dim])
		# (N,L2,hidden_dim)*(N,L2,hidden_dim)=(N,L2,hidden_dim)->(N,hidden_dim)
		f2 = torch.layer_norm(torch.mean(protein2*attn2,dim=1), [self.hidden_dim])

		return f1,f2