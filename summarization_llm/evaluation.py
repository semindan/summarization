#%%
from alignscore import AlignScore

scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='../AlignScoreCheckpoints/AlignScore-base.ckpt', evaluation_mode='nli_sp')
score = scorer.score(contexts=['hello world'], claims=['hello world'])
#%%
score