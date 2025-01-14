from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

from ...utils.molecules import get_molecule_from_data


def filter_valid(data: Data) -> bool:
    """Filter out invalid molecules."""
    # mol = Chem.MolFromSmiles(data.smiles)

    try:
        mol = get_molecule_from_data(data.x, data.edge_index, data.edge_attr)
    except Exception:
        return False

    if len(Chem.DetectChemistryProblems(mol)) != 0:
        return False

    try:
        Descriptors.qed(mol)
    except Exception:
        return False

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False

    try:
        Chem.RemoveHs(mol)
    except Exception:
        print("Error in Chem.RemoveHs")
        return False

    return True
