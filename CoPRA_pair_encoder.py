import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import RNA
from Bio.PDB import PDBParser
from utils.geometry import angstrom_to_nm, pairwise_dihedrals
from models.encoders.layers import AngularEncoding

class AtomSelector(nn.Module):
    def __init__(self, target_atoms: List[str]):
        super().__init__()
        self.target_atoms = target_atoms
        
    def forward(self, pos_atoms: torch.Tensor, mask_atoms: torch.Tensor, atom_names: List[List[str]]):
        """
        Args:
            pos_atoms: (N, L, A, 3) coordinates of the atoms
            mask_atoms: (N, L, A) mask of atoms
            atom_names: (L, A) names of the atoms for each residue
        Returns:
            (N, L, 3) selected positions, (N, L) mask
        """
        device = pos_atoms.device
        N, L, A = pos_atoms.shape[:3]
        
        target_mask = torch.zeros(L, A, dtype=torch.bool, device=device)
        for i in range(L):
            for j in range(A):
                target_mask[i,j] = atom_names[i][j] in self.target_atoms
                
        selected_pos = pos_atoms * target_mask[None,:,:,None]
        selected_mask = mask_atoms * target_mask[None,:,:]
        
        pos_out = torch.zeros(N, L, 3, device=device)
        mask_out = torch.zeros(N, L, device=device)
        
        for n in range(N):
            for i in range(L):
                valid_atoms = selected_mask[n,i].nonzero().flatten()
                if len(valid_atoms) > 0:
                    pos_out[n,i] = selected_pos[n,i,valid_atoms[0]]
                    mask_out[n,i] = 1.0
                    
        return pos_out, mask_out

class RNASecondaryStructureEncoder:
    """Coding of the secondary structure of RNA"""
    
    @staticmethod
    def fold_sequence(sequence: str) -> Dict:
        """Predict structure with ViennaRNA"""
        import RNA
        structure, mfe = RNA.fold(sequence)
        return {
            'structure': structure,
            'mfe': mfe,
            'tensor': torch.tensor([RNASecondaryStructureEncoder._ss_to_code(s) 
                                  for s in structure])
        }
    
    @staticmethod
    def _ss_to_code(symbol: str) -> int:
        """Encoding of secondary structure symbols"""
        code_map = {'(':0, ')':1, '.':2}
        return code_map.get(symbol, 2)

class ProteinGeometryParser:
    """Parse feature strucure from PDB"""
    
    @staticmethod
    def parse_pdb(pdb_path: str) -> Dict:
        """Extracting coordinates of specific atoms"""
        from Bio.PDB import PDBParser
        
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_path)
        
        data = {
            'cbeta': [],
            'cgamma': [],
            'residues': []
        }
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_data = {
                        'name': residue.resname,
                        'atoms': {}
                    }
                    
                    for atom in residue:
                        res_data['atoms'][atom.name] = atom.coord
                        
                    data['residues'].append(res_data)
                    
                    cb = res_data['atoms'].get('CB', None)
                    cg = (res_data['atoms'].get('CG', None) or 
                          res_data['atoms'].get('CG1', None) or
                          res_data['atoms'].get('OG', None))
                    
                    data['cbeta'].append(cb if cb is not None else [0,0,0])
                    data['cgamma'].append(cg if cg is not None else [0,0,0])
                    
        return {
            'cbeta': torch.tensor(data['cbeta']),
            'cgamma': torch.tensor(data['cgamma']),
            'atom_names': [[name for name in res['atoms']] 
                          for res in data['residues']]
        }

class ResiduePairEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=30, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        
        self.aa_pair_embed = nn.Embedding(max_aa_types**2, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)
        self.aapair_to_distcoef = nn.Embedding(max_aa_types**2, max_num_atoms**2)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms**2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU()
        )
        self.dihedral_embed = AngularEncoding()
        
        self.ss_embed = nn.Embedding(3, feat_dim) 
        self.cbeta_selector = AtomSelector(['CB'])
        self.cgamma_selector = AtomSelector(['CG','CG1','OG'])
        
        self.out_mlp = nn.Sequential(
            nn.Linear(feat_dim*5 + self.dihedral_embed.get_out_dim(2), feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def _calc_pairwise_dist(self, pos, mask):
        d = torch.norm(pos[:,:,None] - pos[:,None,:], dim=-1)
        return d * mask[:,:,None] * mask[:,None,:]

    def _calc_relative_pos(self, res_nb, chain_nb):
        same_chain = (chain_nb[:,:,None] == chain_nb[:,None,:])
        relpos = torch.clamp(res_nb[:,:,None] - res_nb[:,None,:], -self.max_relpos, self.max_relpos)
        return self.relpos_embed(relpos + self.max_relpos) * same_chain.float().unsqueeze(-1)

    def _calc_distance_features(self, pos_atoms, mask_atoms, aa):
        d = torch.norm(pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:], dim=-1)
        d = angstrom_to_nm(d.reshape(*d.shape[:3], -1))
        c = F.softplus(self.aapair_to_distcoef(aa[:,:,None]*self.max_aa_types + aa[:,None,:]))
        return self.distance_embed(torch.exp(-c*d**2) * (mask_atoms[:,:,:,None,None] * mask_atoms[:,None,:,None,:]).reshape(*d.shape))

    def _get_pair_mask(self, mask_atoms):
        mask_res = mask_atoms[:,:,1] 
        return mask_res[:,:,None] * mask_res[:,None,:]

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, 
                ss_structure=None, atom_names=None):
        """
        Args:
            ss_structure: (N, L) encoded secondary structure
            atom_names: (L, A) list of atom names for each residue
        """
        N, L = aa.shape
        
        cbeta_pos, cbeta_mask = self.cbeta_selector(pos_atoms, mask_atoms, atom_names)
        cgamma_pos, cgamma_mask = self.cgamma_selector(pos_atoms, mask_atoms, atom_names)
        
        ss_encoded = self.ss_embed(ss_structure) if ss_structure is not None \
            else torch.zeros(N, L, self.ss_embed.embedding_dim, device=aa.device)
        
        d_cbeta = self._calc_pairwise_dist(cbeta_pos, cbeta_mask)
        d_cgamma = self._calc_pairwise_dist(cgamma_pos, cgamma_mask)
        
        feat_aapair = self.aa_pair_embed(aa[:,:,None]*self.max_aa_types + aa[:,None,:])
        feat_relpos = self._calc_relative_pos(res_nb, chain_nb)
        feat_dist = self._calc_distance_features(pos_atoms, mask_atoms, aa)
        feat_dihed = self.dihedral_embed(pairwise_dihedrals(pos_atoms))
        
        feat_all = torch.cat([
            feat_aapair,
            feat_relpos,
            feat_dist,
            self.distance_embed(d_cbeta),
            self.distance_embed(d_cgamma),
            feat_dihed,
            ss_encoded[:,:,None,:].expand(-1,-1,L,-1)
        ], dim=-1)
        
        return self.out_mlp(feat_all) * self._get_pair_mask(mask_atoms)
