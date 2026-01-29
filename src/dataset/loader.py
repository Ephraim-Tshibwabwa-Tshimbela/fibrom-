import torch
from torch.utils.data import Dataset, DataLoader
import yaml

class NF1Dataset(Dataset):
    """Dataset PyTorch pour les données NF1"""
    
    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: Chemin vers les données prétraitées
            split: 'train', 'val', ou 'test'
        """
        self.data = torch.load(f"{data_path}/{split}_data.pt")
        
    def __len__(self):
        return len(self.data['X'])
    
    def __getitem__(self, idx):
        return self.data['X'][idx], self.data['y'][idx]
    
    @property
    def input_size(self):
        return self.data['X'].shape[1]

def create_dataloaders(config_path="config.yml"):
    """Créer les DataLoaders pour train/val/test"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = config['training']['batch_size']
    data_path = config['paths']['processed_data']
    
    # Créer les datasets
    train_dataset = NF1Dataset(data_path, 'train')
    val_dataset = NF1Dataset(data_path, 'val')
    test_dataset = NF1Dataset(data_path, 'test')
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.input_size