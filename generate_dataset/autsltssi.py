"""autsltssi dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from skeleton_graph import tssi_v2
from mediapy import read_video
from extract_landmarks import process_holistic
from autsl_preprocessing import Preprocessing
from pathlib import Path


# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
A large-scale, multimodal dataset that contains isolated Turkish sign videos.
It contains 226 signs that are performed by 43 different signers. There are 36,302 video samples in total.
It contains 20 different backgrounds with several challenges.
"""

# BibTeX citation
_CITATION = """
@article{sincan2020autsl,
  title={AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset and Baseline Methods},
  author={Sincan, Ozge Mercanoglu and Keles, Hacer Yalim},
  journal={IEEE Access},
  volume={8},
  pages={181340--181355},
  year={2020},
  publisher={IEEE}
}
"""


class AutslTssi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for autsltssi dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    TSSI_ORDER = tssi_v2()[1]
    INFO = pd.read_csv("./SignList_ClassId_TR_EN.csv")

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'pose': tfds.features.Tensor(shape=(None, len(self.TSSI_ORDER), 3), dtype=np.float64),
                'label': tfds.features.ClassLabel(names=list(self.INFO["ClassId"].unique().astype(str)))
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('pose', 'label'),  # Set to `None` to disable
            homepage='https://chalearnlap.cvc.uab.cat/dataset/40/description/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples('train'),
            "validation": self._generate_examples('val'),
            "test": self._generate_examples('test')
        }

    def _generate_examples(self, split):
        """Generator of examples for each split."""
        path = Path(f"./pose_{split}")
        labels = pd.read_csv(path / "ground_truth.csv",
                             index_col=0, header=None,
                             names=["filename", "class_id"])

        for filepath in list(path.glob('*.npy')):
            # Yields (key, example)
            filename = filepath.stem
            label = labels.loc[filename]["class_id"]
            yield filename, {
                'pose': self.convert_to_tssi(filepath),
                'label': str(label)
            }
    
    def convert_to_tssi(self, filepath):
        # video_numpy = read_video(filepath)
        # pose = process_holistic(video_numpy)
        pose = np.load(filepath)
        preprocessing_layer = Preprocessing(self.TSSI_ORDER)
        tssi_image = preprocessing_layer(pose)
        return tssi_image.numpy()
