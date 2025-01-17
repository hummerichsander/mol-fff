from typing import Optional

# import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import rdkit
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol


# def generate_svg(molecule: RWMol, size: tuple[int] = (250, 250)) -> str:
#    """Transforms an RDKit molecule object into an SVG image
#
#    :param molecule: RDKit molecule object
#    :param size: size of the image"""
#
#    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
#    d2d.DrawMolecule(molecule)
#    d2d.FinishDrawing()
#    return d2d.GetDrawingText()


# def save_molecule_as(molecule: RWMol, fpath, format: str = "pdf") -> None:
#    """Saves an RDKit molecule object as a file. Supported formats: svg, png, pdf.
#
#    :param molecule: RDKit molecule object
#    :param fpath: path to the file
#    :param format: file format"""
#
#    svg = generate_svg(molecule)
#    if format == "svg":
#        with open(fpath, "w") as f:
#            f.write(svg)
#        return
#    elif format == "png":
#        cairosvg.svg2png(svg, write_to=fpath)
#       return
#    elif format == "pdf":
#        cairosvg.svg2pdf(svg, write_to=fpath)
#        return
#    else:
#       raise ValueError(f"Unknown format: {format}")


def plot_molecule(molecule: RWMol, ax: Optional[plt.Axes] = None):
    """Plots an RDKit molecule object using matplotlib.pyplot.

    :param molecule: RDKit molecule object
    :param ax: matplotlib axis object"""

    img = Draw.MolToImage(molecule)
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")


def plot_molecule_with_validity(mol: RWMol, ax: Optional[plt.Axes] = None):
    """Plots an RDKit molecule object using matplotlib.pyplot. Also shows whether the molecule is
    valid, indicated by a green checkmark or a red cross.

    :param mol: RDKit molecule object
    :param ax: matplotlib axis object"""

    valid = len(rdkit.Chem.DetectChemistryProblems(mol)) == 0
    img = Draw.MolToImage(mol)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img)
    if valid:
        ax.text(
            0.9,
            0.1,
            "\u2713",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=30,
            color="lightgreen",
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.9,
            0.1,
            "\u2717",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=30,
            color="red",
            transform=ax.transAxes,
        )
    ax.axis("off")


def plot_molecule_grid(
    molecules: list[RWMol], ncols: int = 4, box_size: tuple[int] = (2, 2)
) -> plt.Figure:
    """Plots a grid of RDKit molecule objects using matplotlib.pyplot.

    :param molecules: list of RDKit molecule objects
    :param ncols: number of columns in the grid
    :param box_size: size of each box in the grid
    :return: matplotlib figure"""

    rows = int(np.ceil(len(molecules) / ncols))
    fig = plt.figure(figsize=(box_size[0] * ncols, box_size[1] * rows))
    for i, mol in enumerate(molecules):
        ax = fig.add_subplot(rows, ncols, i + 1)
        plot_molecule(mol)
        ax.axis("off")
    plt.tight_layout()
    return fig
