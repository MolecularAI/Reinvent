import rdkit.Chem.Draw as rkcd
import base64
from io import BytesIO


def mol_to_png_string(mol_list: [], molsPerRow=4, subImgSize=(300, 300), legend=None, matches=None):
    image = rkcd.MolsToGridImage(mols=mol_list, molsPerRow=molsPerRow, subImgSize=subImgSize, useSVG=False,
                                 legends=legend, highlightAtomLists=matches)
    buffered = BytesIO()
    image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    buffered.close()
    return str(img_str)[2:][:-1]  # trim on both ends b' and '


def mol_to_svg_string(mol_list: [], molsPerRow=4, subImgSize=(300, 300), legend=None, matches=None):
    image = rkcd.MolsToGridImage(mols=mol_list, molsPerRow=molsPerRow, subImgSize=subImgSize, useSVG=True,
                                 legends=legend, highlightAtomLists=matches)
    return image
