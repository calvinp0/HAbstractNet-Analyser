from rdkit import Chem
from rdkit.Chem import Descriptors, rdchem, rdMolDescriptors
import pandas as pd
import numpy as np

# Define functional groups
functional_groups = {
    'Hydroxyl': '[OX2H]',  # -OH
    'Carbonyl': '[CX3]=[OX1]',  # C=O
    'Amine': '[NX3;H2,H1;!$(NC=O)]',  # -NH2
    'Carboxylic Acid': 'C(=O)[OH]',  # -COOH
    'Ester': 'C(=O)O[C]',  # -COO-
    'Ether': '[OD2]([#6])[#6]',  # R-O-R
    'Aromatic Ring': 'a1aaaaa1',  # Six-membered aromatic ring
    'Alkene': 'C=C',  # C=C double bond
    'Alkyne': 'C#C',  # C≡C triple bond
    'Halogen': '[F,Cl,Br,I]',  # Halogens
    'Nitro': '[$([NX3](=O)=O)]',  # -NO2
    'Sulfonyl': 'S(=O)(=O)[O]',  # -SO2-
    # Add more functional groups as needed
}

def detect_functional_groups(mol, functional_groups):
    fg_counts = {}
    for fg_name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)
        fg_counts[fg_name] = len(matches)
    return fg_counts

def get_atom_features(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        features = {}
        
        atom_type = atom.GetAtomicNum()
        features['atom_type'] = atom_type
        
        features['num_bonds'] = atom.GetDegree()
        
        features['formal_charge'] = atom.GetFormalCharge()
        
        # Chirality
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == rdchem.ChiralType.CHI_UNSPECIFIED:
            features['chirality'] = 'unspecified'
        elif chiral_tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            features['chirality'] = 'tetrahedral_cw'
        elif chiral_tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            features['chirality'] = 'tetrahedral_ccw'
        elif chiral_tag == rdchem.ChiralType.CHI_OTHER:
            features['chirality'] = 'other'
            
        # Number of Hydrogens
        features['num_hydrogens'] = atom.GetTotalNumHs()
        
        # Hybridization
        hybridization = atom.GetHybridization()
        features['hybridization'] = str(hybridization)
        
        # Aromaticity
        features['is_aromatic'] = atom.GetIsAromatic()
        
        # Atomic Mass (divided by 100)
        features['atomic_mass'] = atom.GetMass() / 100.0
        
        atom_features.append(features)
        
    return atom_features

def get_bond_features(mol):
    bond_features = []
    for bond in mol.GetBonds():
        features = {}
        
        # Bond Type
        bond_type = bond.GetBondType()
        if bond_type == rdchem.BondType.SINGLE:
            features['bond_type'] = 'single'
        elif bond_type == rdchem.BondType.DOUBLE:
            features['bond_type'] = 'double'
        elif bond_type == rdchem.BondType.TRIPLE:
            features['bond_type'] = 'triple'
        elif bond_type == rdchem.BondType.AROMATIC:
            features['bond_type'] = 'aromatic'
        else:
            features['bond_type'] = 'unknown'
            
        # Conjugated
        features['is_conjugated'] = bond.GetIsConjugated()
        
        # In Ring
        features['is_in_ring'] = bond.IsInRing()
        
        # Stereo
        stereo = bond.GetStereo()
        if stereo == rdchem.BondStereo.STEREONONE:
            features['stereo'] = 'none'
        elif stereo == rdchem.BondStereo.STEREOANY:
            features['stereo'] = 'any'
        elif stereo == rdchem.BondStereo.STEREOZ:
            features['stereo'] = 'z'
        elif stereo == rdchem.BondStereo.STEREOE:
            features['stereo'] = 'e'
        elif stereo == rdchem.BondStereo.STEREOCIS:
            features['stereo'] = 'cis'
        elif stereo == rdchem.BondStereo.STEREOTRANS:
            features['stereo'] = 'trans'
        else:
            features['stereo'] = 'unknown'
        
        bond_features.append(features)
        
    return bond_features

def aggregate_atom_features(atom_features_list):
    atom_type_counts = {}
    num_bonds_counts = {}
    formal_charge_counts = {}
    chirality_counts = {}
    num_hydrogens_counts = {}
    hybridization_counts = {}
    aromaticity_count = 0
    atomic_masses = []
    
    num_atoms = len(atom_features_list)
    
    for features in atom_features_list:
        
        atom_type = features['atom_type']
        atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1
        
        num_bonds = features['num_bonds']
        num_bonds_counts[num_bonds] = num_bonds_counts.get(num_bonds, 0) + 1
        
        formal_charge = features['formal_charge']
        formal_charge_counts[formal_charge] = formal_charge_counts.get(formal_charge, 0) + 1

        # Chirality Counts
        chirality = features['chirality']
        chirality_counts[chirality] = chirality_counts.get(chirality, 0) + 1

        # Number of Hydrogens Counts
        num_hydrogens = features['num_hydrogens']
        num_hydrogens_counts[num_hydrogens] = num_hydrogens_counts.get(num_hydrogens, 0) + 1

        # Hybridization Counts
        hybridization = features['hybridization']
        hybridization_counts[hybridization] = hybridization_counts.get(hybridization, 0) + 1

        # Aromaticity Count
        if features['is_aromatic']:
            aromaticity_count += 1

        # Atomic Masses
        atomic_masses.append(features['atomic_mass'])

    # Aggregate results into a single dictionary
    aggregated_features = {
        'num_atoms': num_atoms,
        'atom_type_counts': atom_type_counts,
        'num_bonds_counts': num_bonds_counts,
        'formal_charge_counts': formal_charge_counts,
        'chirality_counts': chirality_counts,
        'num_hydrogens_counts': num_hydrogens_counts,
        'hybridization_counts': hybridization_counts,
        'aromatic_atom_count': aromaticity_count,
        'average_atomic_mass': np.mean(atomic_masses) if atomic_masses else None,
    }
    return aggregated_features

def aggregate_bond_features(bond_features_list):
    bond_type_counts = {}
    conjugated_bond_count = 0
    ring_bond_count = 0
    stereo_counts = {}
    
    num_bonds = len(bond_features_list)

    for features in bond_features_list:
        # Bond Type Counts
        bond_type = features['bond_type']
        bond_type_counts[bond_type] = bond_type_counts.get(bond_type, 0) + 1

        # Conjugated Bond Count
        if features['is_conjugated']:
            conjugated_bond_count += 1

        # Ring Bond Count
        if features['is_in_ring']:
            ring_bond_count += 1

        # Stereo Counts
        stereo = features['stereo']
        stereo_counts[stereo] = stereo_counts.get(stereo, 0) + 1

    # Aggregate results into a single dictionary
    aggregated_features = {
        'num_bonds': num_bonds,
        'bond_type_counts': bond_type_counts,
        'conjugated_bond_count': conjugated_bond_count,
        'ring_bond_count': ring_bond_count,
        'stereo_counts': stereo_counts,
    }
    return aggregated_features

def compute_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        descriptors = {
            'MolWt': None,
            'NumHDonors': None,
            'NumHAcceptors': None,
        }
    else:
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        }
    
    return descriptors

def save_descriptors(df, file_path):
    df.to_csv(file_path, index=False)

def flatten_dict(d, parent_key=''):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                new_key = f"{parent_key}{k}_{sub_k}"
                items[new_key] = sub_v
        else:
            new_key = f"{parent_key}{k}"
            items[new_key] = v
    return items

if __name__ == "__main__":
    reactions = pd.read_csv("../data/processed/combined_data.csv")
    
    data = []
    
    for index, row in reactions.iterrows():
        reaction_id = row['reaction_id']
        source = row['source']
        original_reaction_id = row['original_reaction_id']
        
        # Initialize a dictionary to store features for this reaction
        reaction_data = {
            'unique_reaction_id': reaction_id,
            'original_reaction_id': original_reaction_id,
            'source': source
        }
        
        # Process reactants
        for i in range(2):
            reactant_smiles = row[f'reactant_{i}']
            mol = Chem.MolFromSmiles(reactant_smiles)
            if mol:
                # Compute molecular descriptors
                descriptors = compute_molecular_descriptors(reactant_smiles)
                descriptors = {f'reactant_{i}_{key}': value for key, value in descriptors.items()}
                
                fg_counts = detect_functional_groups(mol, functional_groups)
                
                reaction_data[f'reactant_{i}_fg_counts'] = fg_counts
                
                # Get atom and bond features
                atom_features_list = get_atom_features(mol)
                bond_features_list = get_bond_features(mol)
                
                # Aggregate features
                aggregated_atom_features = aggregate_atom_features(atom_features_list)
                aggregated_bond_features = aggregate_bond_features(bond_features_list)
                
                # Extract num_atoms and num_bonds and store them directly
                reaction_data[f'reactant_{i}_num_atoms'] = aggregated_atom_features['num_atoms']
                reaction_data[f'reactant_{i}_num_bonds'] = aggregated_bond_features['num_bonds']
                
                # Remove num_atoms and num_bonds from the aggregated features
                del aggregated_atom_features['num_atoms']
                del aggregated_bond_features['num_bonds']
                
                # Prefix keys to identify reactants
                atom_features_key = f'reactant_{i}_atom_features'
                bond_features_key = f'reactant_{i}_bond_features'
                
                # Update reaction data
                reaction_data.update(descriptors)
                reaction_data[atom_features_key] = aggregated_atom_features
                reaction_data[bond_features_key] = aggregated_bond_features
            else:
                print(f"Invalid SMILES for reactant_{i} in reaction {reaction_id}")
        
        # Process products
        for i in range(2):
            product_smiles = row[f'product_{i}']
            mol = Chem.MolFromSmiles(product_smiles)
            if mol:
                # Compute molecular descriptors
                descriptors = compute_molecular_descriptors(product_smiles)
                descriptors = {f'product_{i}_{key}': value for key, value in descriptors.items()}
                
                # Get atom and bond features
                atom_features_list = get_atom_features(mol)
                bond_features_list = get_bond_features(mol)
                
                # Aggregate features
                aggregated_atom_features = aggregate_atom_features(atom_features_list)
                aggregated_bond_features = aggregate_bond_features(bond_features_list)
                
                # Extract num_atoms and num_bonds and store them directly
                reaction_data[f'product_{i}_num_atoms'] = aggregated_atom_features['num_atoms']
                reaction_data[f'product_{i}_num_bonds'] = aggregated_bond_features['num_bonds']
                
                # Remove num_atoms and num_bonds from the aggregated features
                del aggregated_atom_features['num_atoms']
                del aggregated_bond_features['num_bonds']
                
                # Prefix keys to identify products
                atom_features_key = f'product_{i}_atom_features'
                bond_features_key = f'product_{i}_bond_features'
                
                # Update reaction data
                reaction_data.update(descriptors)
                reaction_data[atom_features_key] = aggregated_atom_features
                reaction_data[bond_features_key] = aggregated_bond_features
            else:
                print(f"Invalid SMILES for product_{i} in reaction {reaction_id}")
        
        # Append the reaction data to the list
        data.append(reaction_data)
    
    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(data)
    
    # Save the DataFrame to a file in CSV format
    save_descriptors(features_df, "../data/descriptors/reaction_descriptors.csv")
