all:
	echo "Nothing"

create:
	mamba create -n llm_env_all_new -y bitsandbytes peft datasets evaluate transformers wandb nltk lightning pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge 
pip_install:
	pip install bitsandbytes
	pip install --upgrade transformers # Install latest version of transformers
	pip install --upgrade accelerate
	pip install --upgrade git+https://github.com/huggingface/peft.git

lit_lightning:
	git clone https://github.com/Lightning-AI/lit-gpt
	cd lit-gpt
	mamba create -n lit
	mamba activate lit
	pip install --index-url https://download.pytorch.org/whl/test/cu118 --pre 'torch==2.1.0'
	pip install -r requirements.txt tokenizers sentencepiece

hf:
	pip install -q huggingface_hub
	pip install -q -U trl transformers accelerate peft
	pip install -q -U datasets bitsandbytes einops wandb
	pip install  -q ipywidgets
	pip install -q scipy
	pip install -q lightning
	pip install -q 'jsonargparse[signatures]'
	pip install -q nltk