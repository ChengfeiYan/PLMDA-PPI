import gvp
import torch.nn

class GraphModule(torch.nn.Module):
	def __init__(self,num_gvp_layers:int,input_dim:int) -> None:
		super().__init__()

		node_input_dim = (input_dim, 50)#2586
		#node_input_dim = (1286, 50)
		edge_input_dim = (432, 25)
		node_hidden_dim = (256, 64)
		edge_hidden_dim = (432, 25)

		input_channel = 0*(node_input_dim[0] + 0*node_input_dim[1])  \
            + 2*(node_hidden_dim[0] + 3*node_hidden_dim[1])  \
            + 1*(144 + 4)
		
		self.embed_node  = torch.nn.Sequential(
			gvp.GVP(node_input_dim, node_hidden_dim,
					activations=(None, None), vector_gate=True),
			gvp.LayerNorm(node_hidden_dim))
		
		self.gvp_conv_layers = torch.nn.ModuleList()
		for i in range(num_gvp_layers):
			self.gvp_conv_layers.append(gvp.GVPConvLayer(node_hidden_dim, 
						edge_hidden_dim,
						n_message=3, 
						n_feedforward=2, 
						drop_rate=0.1,
						vector_gate=True))
			
	def forward(self,
	     nodes1:tuple[torch.Tensor,torch.Tensor],nodes2:tuple[torch.Tensor,torch.Tensor],
		 edges1:tuple[torch.Tensor,torch.Tensor],edges2:tuple[torch.Tensor,torch.Tensor],
		 edge_index1:torch.Tensor,edge_index2:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
		'''
		nodes:((N*L*(6+1280+788+512=2586),(N*L*50*3))#nodes_scat,nodes_vec
		edges:((N*L*432),(N*L*25*3))#edge_scat,edge_vec
		edge_index
		return:((N*L*488),(N*L*488))
		'''
		#GVP not support batch dim
		nodes1 = (nodes1[0].squeeze(0),nodes1[1].squeeze(0))
		nodes2 = (nodes2[0].squeeze(0),nodes2[1].squeeze(0))
		edge_index1 = edge_index1.squeeze(0)
		edge_index2 = edge_index2.squeeze(0)
		edges1 = (edges1[0].squeeze(0),edges1[1].squeeze(0))
		edges2 = (edges2[0].squeeze(0),edges2[1].squeeze(0))

		# print(nodes1[0].shape,nodes1[1].shape)
		# print(nodes2[0].shape,nodes2[1].shape)

		node_emb1 = self.embed_node(nodes1)
		node_emb2 = self.embed_node(nodes2)

		

		for layer in self.gvp_conv_layers:
			node_emb1 = layer(node_emb1,edge_index1,edges1)
			node_emb2 = layer(node_emb2,edge_index2,edges2)

		strucs1,strucv1 = node_emb1
		strucs2,strucv2 = node_emb2

		#(N*L*256+64*3)
		protein1 = torch.cat((strucs1,strucv1.flatten(-2,-1)),dim=-1)
		protein2 = torch.cat((strucs2,strucv2.flatten(-2,-1)),dim=-1)
		protein1 = protein1.unsqueeze(0)
		protein2 = protein2.unsqueeze(0)
		#(N*L*448)

		return protein1,protein2

		