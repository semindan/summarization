import os
from pathlib import Path
import datasets
from summarization_llm.model import LlamaModule
import transformers
import unittest

class TestModel(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = 'meta-llama/Llama-2-13b-chat-hf'

    def test_model_init(self):        
        model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        model = LlamaModule(model_config)
        

if __name__ == '__main__':
    unittest.main()