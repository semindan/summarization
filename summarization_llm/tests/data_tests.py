import unittest
from pprint import pprint
from summarization_llm.llama_data import SumDataset
import summarization_llm
import os
from pathlib import Path
import datasets
import unittest

class TestDataModule(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = 'meta-llama/Llama-2-13b-chat-hf'

    def test_prepare_data(self):
        data = SumDataset()
        data.prepare_data()
        mod_path = Path(__file__).parent.parent
        self.assertTrue(os.path.exists(f"{mod_path}/data"))
        
    def test_build_prompt(self):
        data = SumDataset(overwrite=True)
        data.prepare_data()
        mod_path = Path(__file__).parent.parent
        self.assertTrue(os.path.exists(f"{mod_path}/data"))


if __name__ == '__main__':
    unittest.main()