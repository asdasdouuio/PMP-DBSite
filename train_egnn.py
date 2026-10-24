#!/usr/bin/env python3
# train_egnn.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.spatial.distance import pdist, squareform

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.') 
try:
    from clape.model import StructuralEGNN
except ImportError:
    from model import StructuralEGNN

# ==================== Seed & Setup ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# ==================== 1. Dataset ====================
class ProteinStructureDataset(Dataset):
    def __init__(self, dist_dir, pdb_dir, label_dir, esm2_dir, distance_threshold=10.0, is_train=False, drop_edge_rate=0.15):
        self.dist_dir = dist_dir
        self.pdb_dir = pdb_dir
        self.label_dir = label_dir
        self.esm2_dir = esm2_dir
        self.distance_threshold = distance_threshold
        self.is_train = is_train
        self.drop_edge_rate = drop_edge_rate 
        
        self.file_ids = [os.path.splitext(f)[0] for f in os.listdir(self.dist_dir) if f.endswith('.dist')]
        if not self.is_train: 
            print(f"Found {len(self.file_ids)} protein samples")
            
    def __len__(self):
        return len(self.file_ids)
        
    def _build_edges(self, coords, k_neighbors=15):
        n_nodes = coords.shape[0]
        if n_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 2), dtype=torch.float32)

        dist_matrix = squareform(pdist(coords))
        np.fill_diagonal(dist_matrix, np.inf) 

        k_nearest_idx = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]
        rows = np.repeat(np.arange(n_nodes), k_neighbors)
        cols = k_nearest_idx.flatten()
        distances = dist_matrix[rows, cols]

        valid_mask = distances <= self.distance_threshold
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        valid_distances = distances[valid_mask]

        if len(rows) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 2), dtype=torch.float32)

        spatial_attr = np.exp(-valid_distances / self.distance_threshold)
        seq_distances = np.abs(rows - cols)
        seq_attr = np.exp(-seq_distances / 10.0) 
        
        edges = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
        edge_attr = torch.tensor(np.stack([spatial_attr, seq_attr], axis=1), dtype=torch.float32)

        if self.is_train and self.drop_edge_rate > 0.0 and edges.shape[1] > 0:
            keep_mask = torch.rand(edges.shape[1]) > self.drop_edge_rate
            edges = edges[:, keep_mask]
            edge_attr = edge_attr[keep_mask, :]

        return edges, edge_attr
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        try:
            dist_path = os.path.join(self.dist_dir, f"{file_id}.dist")
            with open(dist_path, 'r') as f:
                sequence = f.read().strip().split('\n')[0]
            seq_len = len(sequence)
            
            pdb_path = os.path.join(self.pdb_dir, f"{file_id}.pdb")
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            model = structure[0]
            
            expected_chain_id = file_id.split('_')[1] if '_' in file_id else 'A'
            chain = model[expected_chain_id] if expected_chain_id in model else next(iter(model.get_chains()), None)

            
            coordinates = []
            backbone_atoms = {'N', 'CA', 'C', 'O', 'OXT'} 
            
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue
                
                sc_mass = 0.0
                sc_com = np.zeros(3)
                
                for atom in residue:
                    name = atom.get_name()
                    if name not in backbone_atoms and atom.element != 'H':
                        mass = atom.mass if hasattr(atom, 'mass') else 12.0 
                        sc_com += atom.get_coord() * mass
                        sc_mass += mass
                
                if sc_mass > 0:
                    coordinates.append(sc_com / sc_mass)
                elif 'CA' in residue:
                    coordinates.append(residue['CA'].get_coord())
                    
            coordinates = np.array(coordinates[:seq_len])

            edges, edge_attr = self._build_edges(coordinates) 
            
            if len(coordinates) < seq_len:
                pad_len = seq_len - len(coordinates)
                coordinates = np.concatenate([coordinates, np.ones((pad_len, 3)) * 1e6], axis=0)
            
            esm2_path = os.path.join(self.esm2_dir, f"{file_id}.npy")
            esm2_embedding = np.load(esm2_path)[:seq_len]
            
            label_path = os.path.join(self.label_dir, f"{file_id}.txt")
            with open(label_path, 'r') as f:
                labels = torch.tensor([int(c) for c in f.read().strip()[:seq_len]], dtype=torch.float32)
            
            return {
                'file_id': file_id, 
                'coordinates': torch.tensor(coordinates, dtype=torch.float32),
                'edges': edges,
                'edge_attr': edge_attr,
                'seq_features': torch.tensor(esm2_embedding, dtype=torch.float32), 
                'labels': labels, 
                'seq_len': seq_len
            }
        except Exception as e:
            print(f"\n[Error] Failed to process protein {file_id}: {str(e)}")
            raise e

def collate_fn(batch):
    batch_size = len(batch)
    max_seq_len = max(item['seq_len'] for item in batch)
    embedding_dim = batch[0]['seq_features'].shape[1]
    
    seq_features_batched = torch.zeros(batch_size, max_seq_len, embedding_dim)
    all_coords, all_labels, all_edges, all_edge_attrs = [], [], [], []
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        coords = item['coordinates']
        if seq_len < max_seq_len:
            pad_coords = torch.full((max_seq_len - seq_len, 3), 1e6)
            coords = torch.cat([coords, pad_coords], dim=0)
        all_coords.append(coords)
        all_labels.append(item['labels'])
        
        offset = i * max_seq_len
        all_edges.append(item['edges'] + offset)
        all_edge_attrs.append(item['edge_attr'])
        
        feat = item['seq_features']
        if seq_len < max_seq_len:
            feat = torch.cat([feat, torch.zeros(max_seq_len - seq_len, embedding_dim)], dim=0)
        seq_features_batched[i] = feat
    
    batched_edges = torch.cat(all_edges, dim=1) if all_edges else torch.empty((2, 0), dtype=torch.long)
    batched_edge_attrs = torch.cat(all_edge_attrs, dim=0) if all_edge_attrs and all_edge_attrs[0].shape[0] > 0 else torch.empty((0, 2), dtype=torch.float32)
    coordinates = torch.stack(all_coords, dim=0).reshape(-1, 3)
    labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-1)
    
    return {
        'seq_features': seq_features_batched, 
        'structural_input': {
            'coordinates': coordinates, 
            'edges': batched_edges,
            'edge_attr': batched_edge_attrs,
            'batch_size': batch_size,
            'mask': labels != -1 
        }, 
        'labels': labels
    }

# ==================== 2. Loss (Focal Loss) ====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=5.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        pos_weight_tensor = torch.tensor([self.pos_weight]).to(inputs.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=pos_weight_tensor
        )
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()

# ==================== 3. Training and Validation ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, accumulation_steps=1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch:03d}')
    
    for batch_idx, batch in enumerate(progress_bar):
        seq_features = batch['seq_features'].to(device)
        structural_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['structural_input'].items()}
        labels = batch['labels'].to(device)
        
        predictions = model(seq_features, structural_input)
        if isinstance(predictions, tuple): predictions = predictions[0]
            
        logits_flat = predictions.reshape(-1)
        mask = labels >= 0
        valid_labels = labels[mask].float()
        valid_logits = logits_flat[mask.flatten()]
        
        loss = criterion(valid_logits, valid_labels)
        current_iter_loss = loss.item() * accumulation_steps
        
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += current_iter_loss
        preds = torch.sigmoid(valid_logits)
        predicted = (preds > 0.5).long()
        
        total += valid_labels.size(0)
        correct += (predicted == valid_labels.long()).sum().item()
        
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total if total > 0 else 0
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            seq_features = batch['seq_features'].to(device)
            structural_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['structural_input'].items()}
            labels = batch['labels'].to(device)
            
            predictions = model(seq_features, structural_input)
            if isinstance(predictions, tuple): predictions = predictions[0]
                
            logits_flat = predictions.reshape(-1)
            mask = labels >= 0
            valid_labels = labels[mask].float()
            valid_logits = logits_flat[mask.flatten()]
            
            loss = criterion(valid_logits, valid_labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(valid_logits)
            predicted = (probs > 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            
            total += valid_labels.size(0)
            correct += (predicted == valid_labels.long()).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    
    print(f"\nValidation Results:")
    print(f"  Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | AUPRC: {auprc:.4f} | AUC-ROC: {auc_roc:.4f}")
    
    return avg_loss, accuracy, auprc, auc_roc

# ==================== 4. Cross-Validation (5-Fold) ====================
def train_one_fold(fold, train_idx, val_idx, config):
    device = torch.device(config['device'])
    print(f"\n>>> Starting Fold {fold+1} (DropEdge: {config['drop_edge_rate']}, Seed: {config['seed']}) <<<")
    
    train_dataset = ProteinStructureDataset(
        config['dist_dir'], config['pdb_dir'], config['label_dir'], 
        config['esm2_dir'], 
        distance_threshold=config['distance_threshold'],
        is_train=True, 
        drop_edge_rate=config['drop_edge_rate'] 
    )
    
    val_dataset = ProteinStructureDataset(
        config['dist_dir'], config['pdb_dir'], config['label_dir'], 
        config['esm2_dir'], 
        distance_threshold=config['distance_threshold'],
        is_train=False,
        drop_edge_rate=0.0 
    )
    
    g = torch.Generator()
    g.manual_seed(config['seed'])

    train_loader = DataLoader(
        Subset(train_dataset, train_idx), 
        batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, 
        num_workers=4, pin_memory=True, generator=g
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx), 
        batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, 
        num_workers=4, pin_memory=True
    )
    
    model = StructuralEGNN(
        hidden_nf=config['hidden_nf'], 
        out_node_nf=1, 
        device=device,
        egnn_layers=2, 
        return_features=False, 
        dropout_rate=config.get('dropout_rate', 0.3),
        input_seq_dim=2560,
        transformer_layers=2,
        transformer_dim=1024 
    ).to(device)
    
    criterion = FocalLoss(gamma=config.get('focal_gamma', 2.0), pos_weight=config.get('pos_weight', 5.0))
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-2))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    
    fold_dir = os.path.join(config['save_dir'], f"egnn_hybrid_seed_{config['seed']}", f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    
    best_auprc, best_epoch = 0.0, 0
    accumulation_steps = config.get('accumulation_steps', 1)
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, accumulation_steps)
        val_loss, val_acc, val_auprc, val_auc_roc = validate_model(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_auprc,
                'val_auc_roc': val_auc_roc,
            }, os.path.join(fold_dir, "best_model.pth"))
            print(f"---  Saved new best model (Validation AUPRC: {best_auprc:.4f}) ---")
            
    print(f"\nFold {fold+1} completed. Best Validation AUPRC: {best_auprc:.4f} at epoch {best_epoch}")
    return best_auprc

# ==================== 5. Main ====================
def main(args):
    print(f"Process Started (Seed Locked to: {args.seed}, DropEdge: {args.drop_edge_rate})")
    base_train_dir = args.data_dir
    
    config = {
        'seed': args.seed, 
        'dist_dir': os.path.join(base_train_dir, 'dist_files'),  
        'pdb_dir': os.path.join(base_train_dir, 'pdb_files'),     
        'label_dir': os.path.join(base_train_dir, 'labels'),      
        'esm2_dir': os.path.join(base_train_dir, 'esm2_features'),
        
        'hidden_nf': 512,
        'distance_threshold': 10.0, 
        'pos_weight': 5.0,      
        'focal_gamma': 2.0,
        
        'batch_size': args.batch_size,
        'accumulation_steps': 8,
        'num_epochs': args.epochs,
        'learning_rate': 2e-4,
        'weight_decay': 1e-2,
        'dropout_rate': 0.3,
        'drop_edge_rate': args.drop_edge_rate, 
        
        'device': args.device,
        'save_dir': './checkpoints_ultimate' 
    }
    
    temp_dataset = ProteinStructureDataset(
        config['dist_dir'], config['pdb_dir'], config['label_dir'], 
        config['esm2_dir'], 
        distance_threshold=config['distance_threshold'], is_train=False
    )
    total_samples = len(temp_dataset)
    del temp_dataset
    
    if total_samples == 0:
        print(f">>> [ERROR] No data found!")
        return 
        
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_auprcs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(total_samples))):
        best_auprc = train_one_fold(fold, train_idx, val_idx, config)
        cv_auprcs.append(best_auprc)
    
    print("\n" + "="*50)
    print(f"5-Fold Cross-Validation Results (Transformer-EGNN Hybrid, Seed {args.seed}):")
    for fold_idx, auprc in enumerate(cv_auprcs):
        print(f"Fold {fold_idx+1}: Best AUPRC = {auprc:.4f}")
    print(f"\nMean AUPRC: {np.mean(cv_auprcs):.4f} (+/- {np.std(cv_auprcs):.4f})")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/train-573')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, help="Set random seed for reproducibility")
    parser.add_argument('--drop_edge_rate', type=float, default=0.15, help="Ratio of edges to randomly drop during training") 
    args = parser.parse_args()
    
    set_seed(args.seed)
    main(args)