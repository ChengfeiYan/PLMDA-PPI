import torch.nn

class ContactModule(torch.nn.Module):
	def __init__(self,block_num:int,input_channels:int,hidden_channels:int,use_paired:bool) -> None:
		super().__init__()
		#whethter use paired features
		self.use_paired = use_paired

		self.input_layer = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=input_channels,
				out_channels=hidden_channels,
				kernel_size=(1,1),
				padding=(0,0),
				dilation=(1,1),
				bias=False),
			torch.nn.InstanceNorm2d(num_features=hidden_channels,
				momentum=0.1,
				affine=True,
				track_running_stats=False),
			torch.nn.LeakyReLU(negative_slope=0.01,
				inplace=False)
		)

		self.hidden_layers = torch.nn.Sequential()
		for i in range(block_num):
			self.hidden_layers.append(OneDimTwoDimBlock(hidden_channels,hidden_channels))

		self.output_layer = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=hidden_channels,
				out_channels=1,
				kernel_size=(1,1),
				padding=(0,0),
				dilation=(1,1),
				bias=False),
			torch.nn.Sigmoid()
		)
	def forward(self,protein1:torch.Tensor,protein2:torch.Tensor)->torch.Tensor:
		'''
		protein1:(N*L1*448)
		protein2:(N*L2*448)
		return:(N*L1*L2)
		'''
		l1 = protein1.shape[-2]
		l2 = protein2.shape[-2]

		protein1 = protein1.unsqueeze(-2)#(N*L1*1*448)
		protein1 = protein1.repeat_interleave(repeats=l2,dim=-2)#(N*L1*L2*448)
		protein2 = protein2.unsqueeze(-3)#(N*1*L2*448)
		protein2 = protein2.repeat_interleave(repeats=l1,dim=-3)#(N*L1*L2*448)

		# z_dif = torch.abs(protein1 - protein2)  # (b, d, N)
		# z_mul = protein1 * protein2
		# protein = torch.cat([z_dif, z_mul], -1)

		# protein = torch.cat((protein1,protein2),dim=-1)#(N*L1*L2*896)

		# protein = protein.transpose(1, 3).transpose(2, 3)#(N*896*L1*L2)

		# protein1 = protein1.unsqueeze(-2)#(N*L1*1*448)
		# protein1 = protein1.repeat_interleave(repeats=l2,dim=-2)#(N*L1*L2*448)
		# protein2 = protein2.unsqueeze(-3)#(N*1*L2*448)
		# protein2 = protein2.repeat_interleave(repeats=l1,dim=-3)#(N*L1*L2*448)

		protein = torch.cat((protein1,protein2),dim=-1)#(N*L1*L2*896)

		protein = protein.transpose(1, 3).transpose(2, 3)#(N*896*L1*L2)

		protein = self.input_layer(protein)#(N*96*L1*L2)
		protein = self.hidden_layers(protein)#(N*96*L1*L2)
		contact_map = self.output_layer(protein)#(N*1*L1*L2)

		

		return contact_map.squeeze(1)#(N*L1*L2)



class OneDimTwoDimBlock(torch.nn.Module):
	def __init__(self,in_channels,out_channels,dilated_rate=1) -> None:
		super().__init__()
		self.conv_3x3 = self.make_conv_layer(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=(3,3),
			padding_size=(1,1),
			dilated_rate=(dilated_rate,dilated_rate))
		
		self.conv_1xn = self.make_conv_layer(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=(1,15),
			padding_size=(0,7*dilated_rate),
			dilated_rate=(1,dilated_rate))
		
		self.conv_nx1 = self.make_conv_layer(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=(15,1),
			padding_size=(7*dilated_rate,0),
			dilated_rate=(dilated_rate,1))
		
		self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
	
	def forward(self,x):
		pass1 = self.conv_3x3(x)
		pass2 = self.conv_1xn(x)
		pass3 = self.conv_nx1(x)

		x = x + pass1 + pass2 + pass3
		x = self.leaky_relu(x)
		return x
	
	def make_conv_layer(self,
			in_channels,
			out_channels,
			kernel_size,
			padding_size,
			dilated_rate=(1,1)):
		layers = torch.nn.Sequential()
		
		layers.append(torch.nn.Conv2d(in_channels=in_channels, 
			out_channels=out_channels, 
			kernel_size=kernel_size,
			padding=padding_size, 
			dilation=dilated_rate, 
			bias=False))
		layers.append(torch.nn.InstanceNorm2d(num_features=out_channels,
			momentum=0.1,
			affine=True, 
			track_running_stats=False))
		layers.append(torch.nn.LeakyReLU(negative_slope=0.01, 
			inplace=False))
		layers.append(torch.nn.Conv2d(in_channels=out_channels, 
			out_channels=out_channels, 
			kernel_size=kernel_size,
			padding=padding_size, 
			dilation=dilated_rate, 
			bias=False))
		layers.append(torch.nn.InstanceNorm2d(num_features=out_channels, 
			momentum=0.1,
			affine=True, 
			track_running_stats=False))

		return layers


