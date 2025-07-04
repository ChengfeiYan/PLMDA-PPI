import torch
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from torchmetrics import Metric

class Distribution(Metric):
	def __init__(self,bins,range) -> None:
		super().__init__()
		self.add_state("true_preds",default=torch.Tensor(),dist_reduce_fx="cat")
		self.add_state("false_preds", default=torch.Tensor(), dist_reduce_fx="cat")
		self.bins = bins

		self.range = range

	def update(self,preds:torch.Tensor,target:torch.Tensor):
		#Convert format
		preds, target = preds.to(self.true_preds), target.to(self.true_preds)
		assert preds.shape == target.shape

		self.true_preds = torch.cat((self.true_preds,preds[target==1]))
		self.false_preds = torch.cat((self.false_preds,preds[target==0]))

	def compute(self):
		'''
		return calculated bin values
		'''
		true_vals = torch.histc(self.true_preds,bins=self.bins,min=min(self.range),max=max(self.range))
		false_vals = torch.histc(self.false_preds,bins=self.bins,min=min(self.range),max=max(self.range))

		return true_vals,false_vals
	
class TopK(Metric):
	def __init__(self,ks=[2,-2]) -> None:
		'''
		ks:topk,negative value will use L/k
		'''
		super().__init__()
		self.ks = ks
		self.add_state("k_acc",default=torch.zeros((len(ks),)),dist_reduce_fx="sum")
		self.add_state("count",default=torch.tensor(0), dist_reduce_fx="sum")
		

	def update(self,preds:torch.Tensor,target:torch.Tensor):
		#Convert format
		#preds, target = preds.to(self.true_preds), target.to(self.true_preds)
		#assert preds.shape == target.shape
		#(N,L1,L2)
		try:
			for i,p in enumerate(preds):
				t = target[i]
				L = (t.shape[0]+t.shape[1])//2
				p = p.flatten()
				t = t.flatten()
				p[t<0]=-1
				for j,k in enumerate(self.ks):
					hit = 0
					if k<0:
						k = L//-k
					values,indices = p.topk(k=k)
					hits = t[indices]
					hits = hits[hits > 0] 
					self.k_acc[j] += len(hits)/k#type:ignore

				self.count += torch.tensor(1)
		except:
			print(target[target>=0].shape)
	def compute(self):
		return self.k_acc/self.count#type:ignore
	

class MapMean(Metric):
	def __init__(self,ks=[2,-2]) -> None:
		'''
		ks:topk,negative value will use L/k
		'''
		super().__init__()
		self.ks = ks
		self.add_state("pred_positive_mean",default=torch.zeros((len(ks),)),dist_reduce_fx="sum")
		self.add_state("pred_negative_mean",default=torch.zeros((len(ks),)),dist_reduce_fx="sum")
		self.add_state("target_positive_mean",default=torch.zeros((len(ks),)),dist_reduce_fx="sum")
		self.add_state("target_negative_mean",default=torch.zeros((len(ks),)),dist_reduce_fx="sum")

		self.add_state("pred_positive_count",default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("pred_negative_count",default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("target_positive_count",default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("target_negative_count",default=torch.tensor(0), dist_reduce_fx="sum")
		

	def update(self,pred_map:torch.Tensor,target_map:torch.Tensor,pred_inter:torch.Tensor,target_inter:torch.Tensor):
		for i,p in enumerate(pred_map):
			t = target_map[i]
			L = (t.shape[0]+t.shape[1])//2
			p = p.flatten()
			t = t.flatten()
			p[t<0]=-1
			for j,k in enumerate(self.ks):
				hit = 0
				if k<0:
					k = L//-k
				values,indices = p.topk(k=k)
				if pred_inter[i] <= 0.5:
					self.pred_negative_mean[j] += values.mean()
				if target_inter[i] == 0:
					self.target_negative_mean[j] += values.mean()
				if pred_inter[i] > 0.5:
					self.pred_positive_mean[j] += values.mean()
				if target_inter[i] == 1:
					self.target_positive_mean[j] += values.mean()


			if pred_inter[i] <= 0.5:
				self.pred_negative_count += torch.tensor(1)
			if target_inter[i] == 0:
				self.target_negative_count += torch.tensor(1)
			if pred_inter[i] > 0.5:
				self.pred_positive_count += torch.tensor(1)
			if target_inter[i] == 1:
				self.target_positive_count += torch.tensor(1)

	def compute(self):
		#4*k -> k*4
		return torch.vstack([
			self.pred_positive_mean/self.pred_positive_count,
			self.pred_negative_mean/self.pred_negative_count,
			self.target_positive_mean/self.target_positive_count,
			self.target_negative_mean/self.target_negative_count])

	
	def plot(self):

		data = self.compute().cpu().numpy()
		data = data.T

		fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(8,8))

		title = ["Mean 5","Mean 10","Mean 50","Mean L/10","Mean L/5","Mean L/2"]
		for i,d in enumerate(data):
			
			x = i%3
			y = i//3
			d = d.reshape(2,2)
			im = ax[y,x].imshow(d,cmap='binary', vmin=0, vmax=0.5)
			ax[y,x].set_title(title[i])
			ax[y,x].set_xticks([0,1])
			ax[y,x].set_xticklabels(['Positive', 'Negative'],rotation = 20)
			ax[y,x].set_yticks([0,1])
			ax[y,x].set_yticklabels(['Prediction', 'Label'],rotation = 20)
			ax[y,x].set_aspect('equal')
			cbar = fig.colorbar(im, ax=ax[y, x], fraction=0.046, pad=0.04)
			cbar.ax.set_position(cbar.ax.get_position().bounds)
			cbar.set_ticks([0, 0.25, 0.5])
			cbar.set_ticklabels(['0', '0.25', '0.5'])

			ax[y, x].tick_params(axis='both', which='major')
			ax[y, x].axhline(0.5, color='red', linestyle='--')

			for m in range(d.shape[0]):
				for n in range(d.shape[1]):
					text = "{:.3f}".format(d[m, n])  # 将数字格式化为小数点后3位
					ax[y, x].text(n, m, text, ha='center', va='center', color='white',path_effects=[withStroke(linewidth=3, foreground='black')],fontsize=8)

		plt.subplots_adjust(wspace=0, hspace=-1)
		plt.suptitle("Contact Map Mean Value of Different Top K Points",y=0.2)
		plt.tight_layout()

		return plt.gcf()