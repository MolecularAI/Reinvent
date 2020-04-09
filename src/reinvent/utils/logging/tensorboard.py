"""
Adapted from tensoroardX.
"""
import io
import torch.utils.tensorboard.summary as tbs
import rdkit.Chem.Draw as rkcd


def add_mol(writer, tag, mol, global_step=None, walltime=None, size=(300, 300)):
    """
    Adds a molecule to the images section of Tensorboard.
    """
    image = rkcd.MolToImage(mol, size=size)
    add_image(writer, tag, image, global_step, walltime)


def add_mols(writer, tag, mols, mols_per_row=1, legends=None, global_step=None, walltime=None, size_per_mol=(300, 300), pattern=None):
    """
    Adds molecules in a grid.
    """
    image = rkcd.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=size_per_mol, legends=legends, highlightAtomLists=pattern)
    add_image(writer, tag, image, global_step, walltime)


def add_image(writer, tag, image, global_step=None, walltime=None):
    """
    Adds an image from a PIL image.
    """
    channel = len(image.getbands())
    width, height = image.size

    output = io.BytesIO()
    image.save(output, format='png')
    image_string = output.getvalue()
    output.close()

    summary_image = tbs.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
    summary = tbs.Summary(value=[tbs.Summary.Value(tag=tag, image=summary_image)])
    writer.file_writer.add_summary(summary, global_step, walltime)
