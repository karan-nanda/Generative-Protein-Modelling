import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

import matplotlib.pyplot as plt

from self_attention import gather_edges, gather_nodes, Normalize

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings,self).__init__()
        
        self.num_embeddings = num_embeddings
        self.period_range = period_range
        
    def forward(self, E_idx):
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)
        ii = torch.arange(N_nodes, dtype = torch.float32).view(1,-1,1)
        d = (E_idx.float() - ii).unsqueeze(-1)
        
        frequency = torch.exp(
            torch.arange(0,self.num_embeddings,2,dtype=torch.float32)
             * -(np.log(10000.0) / self.num_embeddings)
        )
        
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings =16,
                 num_rbf = 16, top_k = 30, features_type ='full', augment_eps=0., dropout=0.1):
        super(ProteinFeatures,self).__init__()
        
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        
        
        #Feature Types
        self.feature_type = features_type
        self.feature_dimensions = {
            'coarse' : (3, num_positional_embeddings + num_rbf + 7),
            'full' : (6, num_positional_embeddings + num_rbf + 7),
            'dist' : (6 , num_positional_embeddings + num_rbf),
            'Hbonds' : (3,2 * num_positional_embeddings),  
        }

        #Positional Encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        
        #Normalization and Embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in, node_features, bias = True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)
        
    def _dist(self, X, mask, eps = 1E-6):
        #Pairwise Euclidean distances on a CNN with NCHW
        
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        
        
        #Identify the K-Nearest Neighbors including self
        
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim =-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        
        return D_neighbors, E_idx, mask_neighbors
    
    def _rbf(self, D):
        #rbf = Radial Distance basis function(similiarity check)
        D_min,D_max,D_count = 0.,20.,self.num_rbf
        D_mu = torch.linspace(D_min,D_max,D_count)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D,-1)
        RBF = torch.exp(-(D_expand - D_mu) / D_sigma) **2

    def _quaternions(self, R):
        #Wikipedia version = en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
          - Rxx + Ryy - Rzz,
          - Rxx - Ryy + Rzz 
        ], -1)))
        
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes #Matmul
        
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        
        
        return Q
    
    def _contacts(self, D_neighbors, E_idx, mask_neighbors, cutoff=8):
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C
    
    def _hbonds(self, X , E_idx, mask_neighbors, eps=1E-3):
        X_atoms = dict(zip(['N', 'CA','C', 'O'], torch.unbind(X,2)))
        
        
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1,:], (0,0,0,1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
            F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
          + F.normalize(X_atoms['N'] - X_atoms['CA'], -1) 
        , -1)
        
        def _distance(X_a, X_b):
            return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)
    
    
        U = (0.084 * 332) * (
                _inv_distance(X_atoms['O'], X_atoms['N'])
            + _inv_distance(X_atoms['C'], X_atoms['H'])
            - _inv_distance(X_atoms['O'], X_atoms['H'])
            - _inv_distance(X_atoms['C'], X_atoms['N'])
        )
        
        
        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
        
        return neighbor_HB
    
    def _orientations_coarse(self,X,E_idx, eps =1e-6):
        
        dX = X[:,1,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        
        #normalize back
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1,u_0), dim=-1)
        
        
        #Bond angles calculations
        cosA  = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA,-1+eps, 1-eps)
        A = torch.acos(cosA)
        #Angling the normalized calcs
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD,-1+eps, 1-eps)
        D = torch.sign((u_2 * u_1).sum(-1)) * torch.acos(cosD)
        
        #backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D),torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0,0,1,2), 'constant',0)
        
        o_1 = F.normalize(u_2 - u_1,dim=-1)
        O = torch.stack((o_1, n_2,torch.cross(o_1,n_2)), 2)
        O = O.view(list(O.shape[:2] + 9))
        O = F.pad(0, (0,0,1,2), 'constant', 0)
        
        
        O_neighbors  = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        O = O.view(list(O.shape[:2] + [3,3]))
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3] )        
        
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2),O_neighbors)
        Q = self._quaternions(R)
        
        O_features = torch.cat((dU,Q), dim=-1)
        
        
        return AD_features, O_features
    
    
    def _dihedrals(self , X,eps = 1e-7):
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1],3)
        
        #Unit vectorization of the shifted slices
        
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)
        
        
          # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features           
    
    def forward(self, X, L, mask):
        
        #Data Augmentation
        if self.training and self.argument_eps > 0:
            X = X + self.augment_eps * torch.rand_like(X)
            
        #K-Nearest neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, E_idx)
        
        #Pairwise Features
        AD_features, O_features = self._orientations_coarse(X_ca,E_idx)
        RBF = self._rbf(D_neighbors)
        
        
        #Pairwise Embeddings
        E_positional = self.embeddings(E_idx)
        
        if self.feature_type == 'coarse':
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
            
        elif self.feature_type == 'hbonds':
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, E_idx, mask_neighbors)
            
            #Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            
            #Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        
        elif self.feature_type == 'full':
            V = self._dihedrals(X)
            E = torch.cat((E_positional,RBF, O_features), -1)
            
        elif self.feature_type == 'dist':
            V = self._dihedrals(X)
            E = torch.cat((E_positional,RBF), -1)
            
        
        #Embedding the nodes
        V = self.node_embedding(V)
        V = self.norm_edges(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        
        
        
        
        return V, E, E_idx
            
            
            
            
         
        

        
        
        
    
    