import os
from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, 'src/version')) as f:
    __version__ = f.read().strip()

install_requires = [
  # verl
  'torch',
  'accelerate',
  'codetiming',
  'datasets>=2.19,<2.21',
  'dill',
  'hydra-core',
  'numpy',
  'pandas',
  'peft',
  'pyarrow>=15,<16',
  'pybind11',
  'pylatexenc',
  'ray[default]>=2.10',
  'tensordict<=0.6.2',
  'torchdata',
  'transformers',
  'vllm==0.8.4',
  'wandb',

  # flashrag
  # 'datasets',
  'base58',
  'nltk',
  'numpy<=1.26.4',
  'langid',
  'openai',
  'peft',
  'PyYAML',
  'rank_bm25',
  'rouge',
  'spacy',
  'tiktoken',
  'tqdm',
  'transformers>=4.40.0',
  'bm25s[core]==0.2.0',
  'fschat',
  'streamlit',
  'chonkie>=0.4.0',
  'gradio>=5.0.0',
  'rouge-chinese',
  'jieba',

  # others
  'sglang',
  'jsonlines',

  # 'flash_attn==2.7.4.post1',
  'hf_transfer'
]

setup(
    name='mentor',
    version=__version__,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    url='https://example.com/anonymous-repo',
    license='MIT License',
    author='Anonymous',
    author_email='anonymous@example.com',
    description='Mentor: Learning to Reason with Tool Call for LLMs via Reinforcement Learning',
    install_requires=install_requires,
    package_data={'': ['**/*.yaml']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
