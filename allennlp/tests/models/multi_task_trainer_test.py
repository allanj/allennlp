"""
This isn't testing a real class, it's a proof-of-concept
for how multi-task training could work. This is certainly
not the only way to do multi-task training using AllenNLP.
Note that you could almost fit this whole setup into
the "SingleTaskTrainer" paradigm, if you just wrote like a
``MinglingDatasetReader`` that wrapped multiple dataset readers.
The main problem is that the ``SingleTaskTrainer`` expects
a single ``train_path``. (Even that you could fudge by passing
in a Dict[str, str] serialized as JSON, but that's really hacky.)
"""
# pylint: disable=bad-continuation

from typing import List, Dict, Iterable, Any, Set
from collections import defaultdict
import os

import tqdm
import torch

from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_pieces import TrainerPieces

@DatasetReader.register("multi-task-test")
class MyReader(DatasetReader):
    """
    Just reads in a text file and sticks each line
    in a ``TextField`` with the specified name.
    """
    def __init__(self, field_name: str) -> None:
        super().__init__()
        self.field_name = field_name
        self.tokenizer = WordTokenizer()
        self.token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokens = self.tokenizer.tokenize(sentence)
        return Instance({self.field_name: TextField(tokens, self.token_indexers)})

    def _read(self, file_path: str):
        with open(file_path) as data_file:
            for line in data_file:
                yield self.text_to_instance(line)



@Model.register("multi-task-test")
class MyModel(Model):
    """
    This model does nothing interesting, but it's designed to
    operate on heterogeneous instances using shared parameters
    (well, one shared parameter) like you'd have in multi-task training.
    """
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.weight = torch.nn.Parameter(torch.randn(()))

    def forward(self,  # type: ignore
                dataset: List[str],
                field_a: torch.Tensor = None,
                field_b: torch.Tensor = None,
                field_c: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        loss = torch.tensor(0.0)   # pylint: disable=not-callable
        if dataset[0] == "a":
            loss += field_a["tokens"].sum() * self.weight
        elif dataset[0] == "b":
            loss -= field_b["tokens"].sum() * self.weight ** 2
        elif dataset[0] == "c":
            loss += field_c["tokens"].sum() * self.weight ** 3
        else:
            raise ValueError(f"unknown dataset: {dataset[0]}")

        return {"loss": loss}




class MultiTaskTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        params = Params({
            "model": {
                "type": "multi-task-test"
                # "type": "multi_task"
            },
            "multi_task": True,
            "iterator": {
                "type": "homogeneous_batch"
            },
            "mingler": {
                "type": "round_robin"
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01
            },
            "dataset_readers": {
                "a": {
                    "type": "multi-task-test",
                    "field_name": "field_a"
                },
                "b": {
                    "type": "multi-task-test",
                    "field_name": "field_b"
                },
                "c": {
                    "type": "multi-task-test",
                    "field_name": "field_c"
                },
            },
            "train_data_paths": {
                "a": self.FIXTURES_ROOT / 'data' / 'babi.txt',
                "b": self.FIXTURES_ROOT / 'data' / 'conll2000.txt',
                "c": self.FIXTURES_ROOT / 'data' / 'conll2003.txt'
            },
            "trainer": {
                "optimizer": {
                    "type":"adam"
                }
            }
        })
        # self.trainer = TrainerBase.from_params(params, self.TEST_DIR)
        import tempfile
        self.serialization_dir = tempfile.mkdtemp()
        pieces = TrainerPieces.from_params(params.duplicate(),  # pylint: disable=no-member
                                           self.serialization_dir,
                                           False,
                                           None,
                                           None,
                                           True)
        self.trainer = Trainer.from_params(model=pieces.model,
                                               serialization_dir=self.serialization_dir,
                                               iterator=pieces.iterator,
                                               train_data=pieces.train_dataset,
                                               validation_data=pieces.validation_dataset,
                                               params=pieces.params,
                                               validation_iterator=pieces.validation_iterator)

    def test_training(self):
        self.trainer.train()

        assert os.path.exists(os.path.join(self.serialization_dir, "best.th"))