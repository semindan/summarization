all:
	echo "Nothing"

create:
	mamba create -n llm_env_all_new -y bitsandbytes peft datasets evaluate transformers wandb nltk lightning pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge 
pip_install:
	pip install bitsandbytes
	pip install --upgrade transformers # Install latest version of transformers
	pip install --upgrade accelerate
	pip install --upgrade git+https://github.com/huggingface/peft.git
