import torch
import torch.nn as nn
from models.metrics import *

class VGGM_nohead(nn.Module):
	def __init__(self, output_dim):
		super(VGGM_nohead, self).__init__()

		# self.class_size = class_size

		# self.conv1 = nn.Sequential(
		# 	nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=0),
		# 	nn.ReLU(),
		# 	nn.BatchNorm2d(96),
		# 	nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
		# )

		# self.conv2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
		# 	nn.ReLU(),
		# 	nn.BatchNorm2d(256),
		# 	nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
		# )

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=1, padding=0),
			nn.ReLU(),
			nn.BatchNorm2d(96),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
		)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=(5,3), stride=(3,2), padding=0, ceil_mode=False)
		)

		# self.fc6 = nn.Sequential(
		# 	nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=1, padding=0),
		# 	nn.ReLU(),
		# 	nn.BatchNorm2d(4096)
		# )
		self.fc6 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(14,1), stride=1, padding=0),
			nn.ReLU(),
			nn.BatchNorm2d(4096)
		)


		self.apool6 = nn.Sequential(
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)

		self.fc7 = nn.Sequential(
			nn.Linear(4096,1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024)
		)

		# self.fc8 = nn.Sequential(
		# 	nn.Linear(1024, self.class_size)
		# )


	def forward(self, x, y):
		x.permute(0,2,1)
		x = x.unsqueeze(1)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.fc6(x)
		x = self.apool6(x)
		x = x.view(-1, 4096)

		x = self.fc7(x)

		return x

class VGGM_simple_head(torch.nn.Module):
    def __init__(self, output_dim, model_settings):
        super(VGGM_simple_head, self).__init__()
        self.backbone = VGGM_nohead(output_dim)
        self.head = nn.Sequential()
        self.head.add_module('linear', nn.Linear(model_settings['emb_size'], output_dim))
    
    def forward(self, x, y):
        out = self.backbone(x, y)
        out = self.head(out)
        return out

class VGGM_arc_margin(torch.nn.Module):
	def __init__(self, output_dim, model_settings):
		super(VGGM_arc_margin, self).__init__()

		self.backbone = VGGM_nohead(output_dim)
		self.head = ArcMarginProduct(model_settings['emb_size'], output_dim, s=model_settings['scale'], m=model_settings['margin'])

	def forward(self, x, y, m, s):
		out = self.backbone(x, y)
		out = self.head(out, y, m, s)
		return out

class VGGM_arc_simple(torch.nn.Module):
	def __init__(self, output_dim, model_settings):
		super(VGGM_arc_simple, self).__init__()

		self.backbone = VGGM_nohead(output_dim)
		self.head1 = ArcMarginProduct(model_settings['emb_size'], output_dim, s=model_settings['scale'], m=model_settings['margin'])
		self.head2 = nn.Sequential()
		self.head2.add_module('linear', nn.Linear(model_settings['emb_size'], output_dim))

	def forward(self, x, y):
		out = self.backbone(x, y)
		arc_logits = self.head1(out, y)
		linear_logits = self.head2(out)
		return arc_logits, linear_logits
