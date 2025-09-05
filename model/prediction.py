import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import TimeSeriesDataset

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

from sklearn.neighbors import NearestNeighbors

# ---------- prediction helpers ----------

def predict_proba(model, X, mask, device=None):
	""" return per-patient probabilities for the 3 tasks """
	with torch.no_grad():
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		ds = TimeSeriesDataset(X, np.zeros((X.shape[0], 3)), mask)
		loader = DataLoader(ds, batch_size=128, shuffle=False)
		model.eval()

		probs_all = []
		for xb, _, _, lb in loader: # Loops over batches
			xb, lb = xb.to(device), lb.to(device)
			z, logits, _ = model(xb, lb)
			probs = torch.sigmoid(torch.stack(logits, dim=1))  # (B,3)
			probs_all.append(probs.cpu())
		return torch.cat(probs_all, dim=0).numpy()

def eval_multitask_from_probs(y_true, probs, plot=True, tr=0.5):
	"""
	y_true, probs: shape (N, 3) — per-patient labels and predicted probabilities.
	Returns dict of metrics per task. If plot=True, draws ROC, PR, and confusion matrix.
	"""
	task_names = ["mortality", "prolonged_stay", "readmission"]
	report = {}

	for t, name in enumerate(task_names):
		yt = y_true[:, t]
		pt_probs = probs[:, t]

		# threshold-free metrics
		roc_auc = roc_auc_score(yt, pt_probs)
		fpr, tpr, thr = roc_curve(yt, pt_probs) # FPR and TPR across thresholds for ROC ploting
		pr_auc  = average_precision_score(yt, pt_probs)
		prec, rec, _  = precision_recall_curve(yt, pt_probs) # precision and recall across thresholds

		# threshold metrics -> needs binary predictions
		pt = (pt_probs >= tr).astype(int)

		acc = accuracy_score(yt, pt) # share of correct predictions
		f1 = f1_score(yt, pt) # balancing false positives/negatives
		ppv = precision_score(yt, pt)

		report[name] = {
			"roc_auc": float(roc_auc),
			"pr_auc": float(pr_auc),
			"precision": float(ppv),
			"accuracy": float(acc),
			"precision": float(ppv),
			"f1": float(f1)
		}

		if plot:
			fig, axes = plt.subplots(1, 3, figsize=(16, 4))

			# ROC
			axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
			axes[0].plot([0, 1], [0, 1], "--", lw=1)
			axes[0].set_title(f"ROC — {name}")
			axes[0].set_xlabel("False Positive Rate")
			axes[0].set_ylabel("True Positive Rate")
			axes[0].legend()
			axes[0].grid(True)

			# PR
			axes[1].plot(rec, prec, label=f"AP = {pr_auc:.3f}")
			axes[1].set_title(f"PR — {name}")
			axes[1].set_xlabel("Recall")
			axes[1].set_ylabel("Precision")
			axes[1].legend()
			axes[1].grid(True)

			# Confusion Matrix
			
			cm = confusion_matrix(yt, pt)
			ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[2], colorbar=False)
			axes[2].set_title(f"Confusion Matrix — {name}")

			plt.tight_layout()
			plt.show()

	return report


def encode_embeddings(model, X, mask, device=None, batch_size=128):
	""" Returns embeddings for each patient """
	with torch.no_grad():
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		model.eval()

		# dummy y just to satisfy the Dataset signature
		ds = TimeSeriesDataset(X, np.zeros((X.shape[0], 3), dtype=np.float32), mask)
		loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

		embs = []
		for xb, _, _, lb in loader:
			xb, lb = xb.to(device), lb.to(device)
			z, _, _ = model(xb, lb) # (B, Z)
			e = model.proj(z) # (B, Zp) -> the SupCon space
			e = F.normalize(e, p=2, dim=1)  # unit-length rows
			embs.append(e.cpu())

		return torch.cat(embs, dim=0).numpy()

def predict_proba_knn(model, X_train, X_test, mask_train, mask_test, n_neig=10, device=None):
	""" return per-patient probabilities for the 3 tasks using KNN in the embeddings """
	# find embeddings (fit index)
	Ztr = encode_embeddings(model, X_train, mask_train)
	Zte = encode_embeddings(model, X_test,  mask_test)

	# cosine on unit sphere; Euclidean works too when normalized
	knn = NearestNeighbors(n_neighbors=n_neig, metric="euclidean")
	knn.fit(Ztr)
	
	# For each test point find its nearest train points
	dist, idx = knn.kneighbors(Zte, n_neighbors=n_neig)
	# simple unweighted vote → mean label among neighbors per task
	return y_train[idx].mean(axis=1)   # shape (N_test, 3)