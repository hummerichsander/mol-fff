import concurrent.futures
from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol


class MolecularMetric(ABC):
    def __call__(self, *args) -> float:
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args) -> float:
        pass


class Validity(MolecularMetric):
    def __init__(self):
        super().__init__()

    def forward(self, molecules: list[RWMol]) -> float:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            valid = list(executor.map(self._get_validity, molecules))
        return sum(valid) / len(molecules)

    def _get_validity(self, mol: RWMol):
        try:
            Chem.SanitizeMol(mol)
            return True
        except ValueError:
            return False


class Uniqueness(MolecularMetric):
    def __init__(self, check_validity: bool = True):
        super().__init__()
        self.check_validity = check_validity

    def forward(self, molecules: list[RWMol]) -> float:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            smiles = list(executor.map(self._get_smiles, molecules))
        smiles = [s for s in smiles if s is not None]
        smiles_unique = set(smiles)
        return len(smiles_unique) / len(smiles)

    def _get_smiles(self, mol: RWMol):
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except ValueError:
            return None


class Novelty(MolecularMetric):
    def __init__(self, reference_molecules: list[RWMol], check_validity: bool = True):
        super().__init__()
        self.reference_molecules = reference_molecules
        self.check_validity = check_validity

    def forward(self, molecules: list[RWMol]) -> float:
        reference_smiles = []
        for mol in self.reference_molecules:
            try:
                Chem.SanitizeMol(mol)
                reference_smiles.append(Chem.MolToSmiles(mol))
            except ValueError:
                continue
        reference_smiles = set(reference_smiles)
        smiles = []
        for mol in molecules:
            try:
                Chem.SanitizeMol(mol)
                smiles.append(Chem.MolToSmiles(mol))
            except ValueError:
                continue
        smiles = set(smiles)
        return len(smiles.difference(reference_smiles)) / len(smiles)


class Components(MolecularMetric):
    def __init__(self, check_validity: bool = True):
        super().__init__()
        self.check_validity = check_validity

    def forward(self, molecules: list[RWMol]) -> float:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            components = list(executor.map(self._get_num_components, molecules))

        components = [c for c in components if c is not None]

        return sum(components) / len(components)

    def distribution(self, molecules: list[RWMol]) -> dict:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            components = list(executor.map(self._get_num_components, molecules))

        components = [c for c in components if c is not None]

        return components

    def _get_num_components(self, mol: RWMol):
        try:
            if self.check_validity:
                Chem.SanitizeMol(mol)
            return len(Chem.GetMolFrags(mol))
        except ValueError:
            return None
