from PIL import Image
from typing import List, Tuple
import numpy as np

from reinvent_chemistry.logging import add_image
from reinvent_chemistry.link_invent.molecule_with_highlighting import MoleculeWithHighlighting
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.transfer_learning.dto.collected_stats_dto import CollectedStatsDTO
from running_modes.transfer_learning.logging.local_transfer_learning_logger import LocalTransferLearningLogger


class LocalLinkInventTransferLearningLogger(LocalTransferLearningLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration=configuration)
        self._molecule_with_highlighting = MoleculeWithHighlighting()

    def _count_compound_frequency_and_track_parts(self, compound_smiles: List[str], part_smiles: List[str])\
            -> Tuple[List, List, List]:
        inchi_dict = dict()

        for smile, smart in zip(compound_smiles, part_smiles):
            mol = self._conversions.smile_to_mol(smile)
            if mol:
                try:
                    inchi_key = self._conversions.mol_to_inchi_key(mol)
                except:
                    self.log_message(f"Failed to transform SMILES string to inchi key: {smile}")
                    continue

                if inchi_key in inchi_dict:
                    inchi_dict[inchi_key]['count'] += 1
                    inchi_dict[inchi_key]['part'].append(smart)
                else:
                    inchi_dict[inchi_key] = {'count': 1, 'mol': mol, 'part': [smart]}

        sorted_list = sorted(inchi_dict.values(), key=lambda x: x['count'], reverse=True)
        list_of_mols = [v['mol'] for v in sorted_list]
        list_of_labels = [f'Times sampled: {v["count"]}' for v in sorted_list]
        list_of_parts = [v['part'] for v in sorted_list]
        return list_of_mols, list_of_parts, list_of_labels,

    def log_time_step(self, epoch: int, learning_rate: float, collected_stats: CollectedStatsDTO,
                      model: GenerativeModelBase):

        if self._with_weights:
            self._weight_stats(model, epoch)

        self._summary_writer.add_scalar('learning_rate', learning_rate, global_step=epoch)

        self._log_nll_stats(epoch, collected_stats)
        self._log_valid_stats(epoch, collected_stats)

        self._visualize_structures_highlighted(epoch, 'training', collected_stats.training_stats.molecule_smiles,
                                               collected_stats.training_stats.molecule_parts_smiles)
        if collected_stats.validation_nll:
            self._visualize_structures_highlighted(epoch, 'validation',
                                                   collected_stats.validation_stats.molecule_smiles,
                                                   collected_stats.validation_stats.molecule_parts_smiles)

    def _log_nll_stats(self, epoch: int, stats: CollectedStatsDTO):
        self._summary_writer.add_histogram("nll_plot/training", np.array(stats.nll), global_step=epoch)
        self._summary_writer.add_histogram("nll_plot/sampled_training",
                                           np.array(stats.training_stats.nll_input_sampled_target), global_step=epoch)
        nll_avg = {
            "training": np.mean(stats.nll),
            "sampled_training": np.mean(stats.training_stats.nll_input_sampled_target)
        }
        nll_var = {
            "training": np.var(stats.nll),
            "sampled_training": np.var(stats.training_stats.nll_input_sampled_target)
        }

        if stats.validation_nll is not None:
            self._summary_writer.add_histogram("nll_plot/validation", np.array(stats.validation_nll), global_step=epoch)
            self._summary_writer.add_histogram("nll_plot/sampled_validation",
                                               np.array(stats.validation_stats.nll_input_sampled_target),
                                               global_step=epoch)
            nll_avg["validation"] = np.mean(stats.validation_nll)
            nll_avg["sampled_validation"] = np.mean(stats.validation_stats.nll_input_sampled_target)
            nll_var["validation"] = np.var(stats.validation_nll)
            nll_var["sampled_validation"] = np.var(stats.validation_stats.nll_input_sampled_target)

        self._summary_writer.add_scalars("nll/avg", nll_avg, global_step=epoch)
        self._summary_writer.add_scalars("nll/var", nll_var, global_step=epoch)

        self._summary_writer.add_scalar("nll_plot/jsd_binned", stats.jsd_binned, global_step=epoch)
        self._summary_writer.add_scalar("nll_plot/jsd_un_binned", stats.jsd_un_binned, global_step=epoch)

    def _log_valid_stats(self, epoch: int, stats: CollectedStatsDTO):
        self._summary_writer.add_scalar('valid_compounds/training', stats.training_stats.valid_fraction,
                                        global_step=epoch)
        if stats.validation_stats is not None:
            self._summary_writer.add_scalar('valid_compounds/validation', stats.validation_stats.valid_fraction,
                                            global_step=epoch)

    def _visualize_structures_highlighted(self, epoch, tag, compound_smiles_list: List[str],
                                          joined_parts_smiles_list: List[str]):

        count = 0
        img_list = []
        for mol, parts_list, label in zip(*self._count_compound_frequency_and_track_parts(
                compound_smiles_list, joined_parts_smiles_list)):
            try:
                img_list.append(self._molecule_with_highlighting.get_image(mol, parts_list, label))
                count += 1
            except Exception as e:
                self.log_message(f'Image creation failed with error \n\t{str(e)}')
                continue
            if count == self._sample_size:
                break

        if img_list:  # there are valid molecules and the plotting was successful
            grid_image = self._img_list_to_grid_image(img_list)
            add_image(self._summary_writer, f"{tag}/Most Frequent Molecules",  grid_image, global_step=epoch)
        else:
            self.log_message(f'No molecules to show for epoch {epoch} - {tag}')

    def _img_list_to_grid_image(self, image_list: List[Image.Image]):
        if image_list:
            w, h = image_list[0].size
            grid = Image.new('RGB', size=(self._columns * w, self._rows * h), color=(255, 255, 255))
            for i, image in enumerate(image_list):
                grid.paste(image, box=(i % self._columns * w, i // self._columns * h))
        else:
            grid = Image.new('RGB', size=(100, 100), color=(255, 255, 255))
        return grid
