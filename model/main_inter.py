import torch,os,sys,comet_ml
sys.path.append(os.getcwd())

import torch,torch.nn

from model.graph_module import GraphModule
from model.contact_module import ContactModule
from model.interaction_module import InteractionModule
from model.graph_sub_module import GraphSubModule

import metrics
import lightning.pytorch as pl
import torchmetrics.classification as mc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class MainModel(pl.LightningModule):
	def __init__(self,hyper_params:dict) -> None:
		super().__init__()
		self.hyper_params = hyper_params
		#Modules
		if "use_sub" in self.hyper_params["model"]["graph_module"] and self.hyper_params["model"]["graph_module"]["use_sub"]:
			self.graph_module = GraphSubModule(
				input_dim=get_input_dim(self.hyper_params["dataset"]["settings"]["features"]),hidden_dim=512,out_dim=448,depth=3)
		else:
			self.graph_module = GraphModule(
				num_gvp_layers=self.hyper_params["model"]["graph_module"]["num_gvp_layers"],
				input_dim=get_input_dim(self.hyper_params["dataset"]["settings"]["features"])
				)

		self.contact_module = ContactModule(
				block_num=self.hyper_params["model"]["contact_module"]["block_num"],
				input_channels=self.hyper_params["model"]["contact_module"]["input_channels"],
				hidden_channels=self.hyper_params["model"]["contact_module"]["hidden_channels"],
				use_paired="pair_features" in self.hyper_params["dataset"]["settings"]["features"])
		self.interaction_module = InteractionModule(
			input_dim=self.hyper_params["model"]["interaction_module"]["input_dim"],
			hidden_dim=self.hyper_params["model"]["interaction_module"]["hidden_dim"],
			dma_depth=self.hyper_params["model"]["interaction_module"]["dma_depth"])

		#Loss
		self.loss = torch.nn.BCELoss()

		#Metrics
		self.acc = mc.BinaryAccuracy(0.5)
		self.f1 = mc.BinaryF1Score(0.5)
		self.roc = mc.BinaryROC()
		self.auroc = mc.BinaryAUROC()
		self.pr = mc.BinaryPrecisionRecallCurve()
		self.dist = metrics.Distribution(10,(0.,1.))
		self.topk = metrics.TopK(ks=[5,10,50,-10,-5,-2])
		self.map_mean = metrics.MapMean(ks=[5,10,50,-10,-5,-2])

		#self.automatic_optimization = False

	def forward(self,input)->list[torch.Tensor]:
		#extract input
		nodes1, edges1, edge_index1 , nodes2, edges2, edge_index2,emb1,emb2 = input
		#calculating
		protein1,protein2 = self.graph_module(nodes1,nodes2,edges1,edges2,edge_index1,edge_index2)

		#protein1 = self.dropout(protein1)
		#protein2 = self.dropout(protein2)

		contact_map = self.contact_module(protein1,protein2)
		
		interaction = self.interaction_module(emb1,emb2,contact_map)
		
		return [contact_map,interaction] 
	

	def training_step(self, batch, batch_idx):
		input,[label,contact_map,interaction] = batch


		#Do prediction
		pred_map,pred_inter = self(input)
		contact_map = contact_map.to(pred_map)
		interaction = interaction.to(pred_inter)
		print(contact_map.shape)
		#Contact mask
		pred_copy = pred_map.detach()
		contact_copy = contact_map.detach()
		pred_map = pred_map[contact_map>=0]
		contact_map = contact_map[contact_map>=0]


		inter_loss = self.loss(pred_inter, interaction)
		map_loss = self.loss(pred_map, contact_map)

		loss = map_loss * self.hyper_params["ratios"][0] + inter_loss * self.hyper_params["ratios"][1]

		
		
		self.log('train_map_loss', map_loss,prog_bar=True,on_epoch=True,on_step=True,logger=True)

		if self.hyper_params["ratios"][1] != 0:
			self.log('train_inter_loss', inter_loss,prog_bar=True,on_epoch=True,on_step=True,logger=True)


		#log samples
		if self.global_step % 500 == 0:
			#if self.hyper_params["ratios"][0] != 0:
			plt.cla()
			plt.clf()
			plt.subplot(121)
			plt.imshow(pred_copy[0].cpu(), cmap='bwr',
					vmin=-1, vmax=1, interpolation='none')
			plt.colorbar()
			plt.subplot(122)
			plt.imshow(contact_copy[0].cpu(), cmap='bwr',
					vmin=-1, vmax=1, interpolation='none')
			plt.title(str(len(contact_map[contact_map > 0])))
			plt.colorbar()
			plt.tight_layout()
			self.logger.experiment.log_figure(
				"sample_map.png", plt.gcf(), step=self.global_step)
			plt.clf()
			plt.cla()
				

		return {'loss':loss}

		

	def validation_step(self, batch, batch_idx):
		input = batch
		pred_map,pred_inter = self(input)

		contact_map = contact_map.to(pred_map)
		interaction = interaction.to(pred_inter)

		#Contact mask
		mask = contact_map>=0
		pred_map_mask = pred_map[mask]
		contact_map_mask = contact_map[mask]
		
		

		map_loss = self.loss(pred_map_mask, contact_map_mask)
		inter_loss = self.loss(pred_inter, interaction)
		

		#loss = self.hyper_params["ratios"][0]*map_loss+self.hyper_params["ratios"][1]*inter_loss


		#covert true value to int and log preds
		interaction = interaction.to(torch.int)

		#Update metrics
		self.acc.update(pred_inter,interaction)
		self.f1.update(pred_inter,interaction)
		self.roc.update(pred_inter, interaction)
		self.auroc.update(pred_inter, interaction)
		self.pr.update(pred_inter, interaction)
		self.dist.update(pred_inter,interaction)
		if int(interaction.data):#only record true maps
			self.topk.update(pred_map,contact_map)

		self.map_mean.update(pred_map,contact_map,pred_inter,interaction)

		#Log val loss
		#self.log('val_loss', loss, prog_bar=True,on_epoch=True,on_step=False,logger=True)
		if self.hyper_params["ratios"][0] != 0:
			self.log('val_map_loss', map_loss,on_epoch=True,on_step=False,logger=True)
		if self.hyper_params["ratios"][1] != 0:
			self.log('val_inter_loss', inter_loss,on_epoch=True,on_step=False,logger=True)
		
		self.log('val_loss', map_loss*self.hyper_params["ratios"][0]+inter_loss*self.hyper_params["ratios"][1],on_epoch=True,on_step=False,logger=True)


		
		#return {'val_loss': loss, 'val_inter_loss': inter_loss, 'val_map_loss': map_loss, 'pred_map': pred_map, 'pred_inter': pred_inter, 'contact_map': contact_map, 'interaction': interaction}

	def on_validation_epoch_end(self) -> None:
		#assert isinstance(self.logger.experiment, cm.Experiment)
		#Accuracy
		self.log(f'acc',self.acc.compute(),on_epoch=True,on_step=False,logger=True)
		self.acc.reset()

		#F1 score
		self.log(f'f1',self.f1.compute(),on_epoch=True,on_step=False,logger=True)
		self.f1.reset()
		
		#ROC curve
		fpr,tpr,thres = self.roc.compute()
		plt.plot(fpr.cpu(), tpr.cpu())
		plt.title("ROC Curve")
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.grid(True, linestyle='--', linewidth=0.5,alpha=0.5)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.tight_layout()

		self.logger.experiment.log_figure("roc.png",plt.gcf(),step=self.global_step)
		plt.clf()
		plt.cla()
		self.logger.experiment.log_curve("roc",fpr.tolist(),tpr.tolist(),step=self.global_step)
		self.roc.reset()

		#AUROC
		self.log('auroc',self.auroc.compute(),on_epoch=True, on_step=False, logger=True)
		self.auroc.reset()
		
		#PR curve
		p,r,thres = self.pr.compute()
		plt.plot(p.cpu(), r.cpu())
		plt.plot([0, 1], linewidth=0.5, alpha=1, linestyle='--')
		plt.title("PR Curve")
		plt.xlabel("Precision")
		plt.ylabel("Recall")
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.grid(True, linestyle='--', linewidth=0.5,alpha=0.5)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.tight_layout()

		self.logger.experiment.log_figure("pr.png",plt.gcf(),step=self.global_step)
		plt.clf()
		plt.cla()
		self.logger.experiment.log_curve("pr",p.tolist(),r.tolist(),step=self.global_step)
		self.pr.reset()

		#AUPR
		self.log('aupr',torch.abs(torch.trapezoid(y=p,x=r)),on_epoch=True, on_step=False)

		#Result Distribution
		true,false = self.dist.compute()

		plt.grid(True, axis='y', linestyle='--', linewidth=0.5,alpha=0.5)

		plt.bar(torch.linspace(min(self.dist.range),max(self.dist.range),self.dist.bins+1)[:-1], true.cpu(), width=0.1, align='edge', edgecolor='black', linewidth=1, label='Positive')
		plt.bar(torch.linspace(min(self.dist.range),max(self.dist.range),self.dist.bins+1)[:-1], -false.cpu(), width=0.1, align='edge', edgecolor='black', linewidth=1, label='Negative')
		ax = plt.gca()
		ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(abs(x))))
		ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

		plt.xlim(0, 1)
		plt.title("Result distribution")
		plt.xlabel("Predicted Interaction Probability")
		plt.ylabel("Count")
		plt.legend()
		plt.xlim(0,1)
		plt.tight_layout()

		self.logger.experiment.log_figure("dist.png", plt.gcf(), step=self.global_step)
		plt.clf()
		plt.cla()
		self.logger.experiment.log_histogram_3d(true.tolist(
		), "true_distribution", step=self.global_step, epoch=self.current_epoch)
		self.logger.experiment.log_histogram_3d(false.tolist(
		), "false_distribution", step=self.global_step, epoch=self.current_epoch)
		self.dist.reset()

		#Topk
		top5,top10,top50,topL10,topL5,topL2 = self.topk.compute()
		self.log_dict({"top5":top5,"top10":top10,"top50":top50,"topL10":topL10,"topL5":topL5,"topL2":topL2},on_epoch=True,on_step=False,logger=True)
		self.topk.reset()

		#Meank
		means = self.map_mean.compute()
		
		self.logger.experiment.log_figure("mean.png", self.map_mean.plot(), step=self.global_step)
		plt.clf()
		plt.cla()
		

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(
		), lr=self.hyper_params["learning_rate"], weight_decay=self.hyper_params["weight_decay"], amsgrad=True)

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.1,patience=3,eps=1e-8,threshold=1e-4,threshold_mode='rel',verbose=True)

		return {'optimizer':optimizer,'lr_scheduler':scheduler,"monitor":"val_inter_loss"}
	

	def optimizer_step(self,epoch:int,batch_idx:int,optimizer,optimizer_closure = None) -> None:
		for param_group in optimizer.param_groups:#type:ignore
			self.log('learning_rate',param_group['lr'],on_epoch=True, on_step=False, logger=True)
		return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

############################################################################################################
def get_input_dim(features:list)->int:
	#init node dim in graph
	dim = 6
	if "msa_emb" in features:
		dim += 768
	if "pssm" in features:
		dim += 20
	if "structure_emb" in features or "structure_emb_af" in features:
		dim += 512

	return dim
