import concurrent.futures
from typing import Optional, Literal

from rdkit import Chem
from rdkit.Chem import RWMol
from torch import Tensor
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData
from torch_geometric.utils import to_undirected


def geometric_to_mol(
        graph: GeometricData | GeometricBatch | list[GeometricData],
        max_workers: Optional[int] = None,
) -> list[RWMol]:
    """Transforms a PyG graph or batch of graphs into a list of RDKit molecules
    :param graph: The graph instance or batch of graphs.
    :param max_workers: The maximum number of workers to use for parallel processing.
    :return: The RDKit molecule or a list of RDKit molecules."""

    def get_molecule(data: GeometricData) -> RWMol:
        mol = Chem.RWMol()
        for atomic_number in data.x:
            # one-hot encoded, take (argmax + 1) to get atomic number
            if len(atomic_number) != 1:
                atomic_number = int(atomic_number.argmax(dim=-1) + 1)
            mol.AddAtom(Chem.Atom(atomic_number))

        for edge, bond_type in zip(data.edge_index.T, data.edge_attr):
            if edge[0] >= edge[1]:
                continue
            bond_type = bond_from_one_hot(bond_type, output_type="chem")
            if bond_type:
                mol.AddBond(
                    int(edge[0]),
                    int(edge[1]),
                    bond_type,
                )
        return mol

    if isinstance(graph, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            return list(executor.map(get_molecule, graph))

    elif hasattr(graph, "batch") and graph.batch is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            return list(executor.map(get_molecule, graph.to_data_list()))

    else:
        return [get_molecule(graph)]


def atom_from_one_hot(
        one_hot: Tensor, profile: Literal["qm9", "zinc", "pcqm4m", "unimers"]
) -> Chem.Atom:
    """Converts a one-hot encoded atom type to Chem.Atom

    :param one_hot: One-hot encoded atom type
    :param profile: Atom profile to use
    :return: rdkit Atom object"""

    match profile:
        case "pcqm4m":
            atom_mapping = {
                0: "H",
                1: "C",
                2: "Li",
                3: "Be",
                4: "B",
                5: "C",
                6: "N",
                7: "O",
                8: "F",
                9: "Ne",
                10: "Na",
                11: "Mg",
                12: "Al",
                13: "Si",
                14: "P",
                15: "S",
                16: "Cl",
                17: "Ar",
                18: "K",
                19: "Ca",
                20: "Sc",
                21: "Ti",
                22: "V",
                23: "Cr",
                24: "Mn",
                25: "Fe",
                26: "Co",
                27: "Ni",
                28: "Cu",
                29: "Zn",
                30: "Ga",
                31: "Ge",
                32: "As",
                33: "Se",
                34: "Br",
                35: "Kr",
            }
        case "qm9":
            atom_mapping = {
                0: "H",
                1: "He",
                2: "Li",
                3: "Be",
                4: "B",
                5: "C",
                6: "N",
                7: "O",
                8: "F",
            }
        case "zinc":
            atom_mapping = {
                0: "C",
                1: "O",
                2: "N",
                3: "F",
                4: "C H1",
                5: "S",
                6: "Cl",
                7: "O -",
                8: "N H1 +",
                9: "Br",
                10: "N H3 +",
                11: "N H2 +",
                12: "N +",
                13: "N -",
                14: "S -",
                15: "I",
                16: "P",
                17: "O H1 +",
                18: "N H1 -",
                19: "O +",
                20: "S +",
                21: "P H1",
                22: "P H2",
                23: "C H2 -",
                24: "P +",
                25: "S H1 +",
                26: "C H1 -",
                27: "P H1 +",
            }
        case "unimers":
            atom_mapping = {
                0: "C",
                1: "N",
                2: "O",
                3: "F",
            }
        case other:
            raise ValueError(f"Unknown atom profile: {other}")

    atom_type = int(one_hot.argmax().item())

    if atom_type not in atom_mapping:
        raise ValueError(f"Unknown atom type: {atom_type}")

    atom_label = atom_mapping[atom_type]
    elements = atom_label.split()
    atom = Chem.Atom(elements[0])

    for el in elements[1:]:
        if el.startswith("H"):
            atom.SetNumExplicitHs(int(el[1:]))
        elif el == "+":
            atom.SetFormalCharge(1)
        elif el == "-":
            atom.SetFormalCharge(-1)
        elif el.startswith("+"):
            atom.SetFormalCharge(int(el[1:]))
        elif el.startswith("-"):
            atom.SetFormalCharge(-int(el[1:]))

    return atom


def bond_from_one_hot(
        one_hot: Tensor, profile: Literal["qm9", "zinc", "pcqm4m", "unimers"] = "qm9"
) -> Chem.BondType:
    """Converts a one-hot encoded bond type to Chem.BondType

    :param one_hot: One-hot encoded bond type
    :param profile: Bond profile to use
    :return: rdkit BondType object"""

    match profile:
        case "pcqm4m":
            bond_types = {
                0: Chem.BondType.SINGLE,
                1: Chem.BondType.DOUBLE,
                2: Chem.BondType.TRIPLE,
                3: Chem.BondType.AROMATIC,
                4: None,
            }
        case "qm9":
            bond_types = {
                0: Chem.BondType.SINGLE,
                1: Chem.BondType.DOUBLE,
                2: Chem.BondType.TRIPLE,
                3: Chem.BondType.AROMATIC,
                4: None,
            }
        case "zinc":
            bond_types = {
                0: Chem.BondType.SINGLE,
                1: Chem.BondType.DOUBLE,
                2: Chem.BondType.TRIPLE,
                3: None,
            }
        case "unimers":
            bond_types = {
                0: Chem.BondType.SINGLE,
                1: Chem.BondType.DOUBLE,
                2: Chem.BondType.TRIPLE,
                3: None
            }
        case other:
            raise ValueError(f"Unknown bond profile: {other}")

    bond_type = int(one_hot.argmax().item())

    if bond_type not in bond_types:
        raise ValueError(f"Unknown bond type: {bond_type}")

    return bond_types[bond_type]


def get_molecule_from_data(
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        profile: Literal["qm9", "zinc", "pcqm4m", "unimers"] = "qm9",
) -> Chem.RWMol:
    """Converts a PyTorch Geometric graph to an RDKit molecule

    :param x: Node features (one-hot encoded atom types)
    :param edge_index: Edge indices
    :param edge_attr: Edge features (one-hot encoded bond types)
    :param profile: Atom and bond profile to use
    :return: RDKit molecule"""
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce="mean")

    mol = Chem.RWMol()

    for one_hot in x:
        mol.AddAtom(atom_from_one_hot(one_hot, profile))

    for edge, one_hot in zip(edge_index.T, edge_attr):
        if edge[0] >= edge[1]:
            continue
        bond_type = bond_from_one_hot(one_hot, profile=profile)
        if bond_type is not None:
            mol.AddBond(int(edge[0]), int(edge[1]), bond_type)

    return mol
