import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ---------- Model ----------
class MultiTaskSeqGRUAE(nn.Module):
	def __init__(self, input_dim, enc_hidden=128, enc_layers=1, dec_hidden=128, dec_layers=1, latent_dim=64, SupCon_latent_dim=32,dropout=0.1):
		""" input_dim : number of features per timestep (D)
		   enc_hidden: hidden size of the encoder GRU
		   dec_hidden: hidden size of the decoder GRU
		   enc_layers: num of stacked GRU layers in the encoder
		   dec_layers: num of stacked GRU layers in the decoder
		   latent_dim: size of the shared patient latent vector z 
		"""
		super().__init__()
		
		# encoder: GRU reads X's timeline step-by-step and compresses it into latent space
		self.encoder = nn.GRU(input_dim, enc_hidden, enc_layers, batch_first=True, dropout=dropout if enc_layers > 1 else 0.0, bidirectional=False)

		# turn the encoder’s hidden vector into latent size
		self.to_latent = nn.Linear(enc_hidden, latent_dim)

		# projection head for contrastive learning - 2-layer MLP for each task 
		self.proj_heads = nn.ModuleList([
			nn.Sequential(
				nn.Linear(latent_dim, SupCon_latent_dim), 
				nn.ReLU(inplace=True), 
				nn.Linear(SupCon_latent_dim, SupCon_latent_dim)) 
			for _ in range(3)])

		# three classifiers (separate linear heads - one per task) from shared latent z
		self.cls_heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(3)])

		# turn the patient representation in z (B, latent_dim) into the decoder’s start state
		self.z_to_h0 = nn.Linear(latent_dim, dec_layers * dec_hidden)
		
		# decoder: reconstruct X from z with GRU -> from vector per patient, rebuild the whole sequence X̂ (B, T, D)
		self.decoder = nn.GRU(input_size=latent_dim, hidden_size=dec_hidden, num_layers=dec_layers, batch_first=True, dropout=dropout if dec_layers > 1 else 0.0)
		
		# reconstruct original features from GRU hidden state 
		self.out = nn.Linear(dec_hidden, input_dim)

	def encode(self, x, lengths):
		packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
		_, hN = self.encoder(packed) # the final hidden states for each layer(num_layers, B, H)
		# last layer’s final hidden state - summary of each patient’s whole sequence
		h_last = hN[-1] # (B, H)
		z = self.to_latent(h_last) # (B, Z)
		return z

	def decode(self, z, T):
		""" z: patient embedding for the batch (B, latent_dim)
		    T: number of timesteps to reconstruct """
		# repeat z across time
		B, Z = z.shape
		z_seq = z.unsqueeze(1).repeat(1, T, 1)  # (B,T,Z) with copies of z at every timestep (keeps a constant patient fingerprint at each step)
		
		# map z to the GRU’s initial hidden state
		h0 = self.z_to_h0(z).view(self.decoder.num_layers, B, self.decoder.hidden_size).contiguous()
		# run decoder GRU over time
		dec_out, _ = self.decoder(z_seq, h0) # (B,T,Hd)
		# linear layer converts each timestep’s hidden state to D features
		x_hat = self.out(dec_out) # (B,T,D)
		return x_hat

	def forward(self, x, lengths):
		B, T, _ = x.shape
		z = self.encode(x, lengths)
		# logits per task - 3 × (B,)
		logits = [head(z).squeeze(-1) for head in self.cls_heads] 
		x_hat = self.decode(z, T)
		return z, logits, x_hat
		
	def project(self, z, k):
		e = self.proj_heads[k](z)
		# L2 normalize keeps only “shape/direction” -> so patients with similar outcomes cluster together even if their raw feature scales differ
		return F.normalize(e, p=2, dim=1)
