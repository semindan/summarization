import os
from pathlib import Path
import datasets
from summarization_llm.llama import LlamaModule
from summarization_llm.llama_data import SumDataset
import transformers
import unittest

class TestPipeline(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = 'meta-llama/Llama-2-13b-chat-hf'

    def test_one_train_step(self):        
        model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        data = SumDataset(overwrite=True)
        data.setup("fit")
        model = LlamaModule(model_config)
        loader = data.train_dataloader()
        for batch in loader:
            out = model.training_step(batch, 0)
            print("out:", out)
            break
        

if __name__ == '__main__':
    unittest.main()