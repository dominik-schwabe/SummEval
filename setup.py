
from setuptools import setup
from setuptools import find_packages


setup(name='summ_eval',
      version='0.1',
      description='Toolkit for summarization evaluation',
      url='https://github.com/Alex-Fabbri/summ_eval.git',
      author='Alex Fabbri, Wojciech Kryściński',
      author_email='alexander.fabbri@yale.edu, wojciech.kryscinski@salesforce.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      dependency_links=[
        "git://github.com/bheinzerling/pyrouge.git#egg=pyrouge",
        "git://github.com/dominik-schwabe/emnlp19-moverscore.git#egg=moverscore",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.2.0/en_core_web_md-2.2.0.tar.gz#egg=en_core_web_md",
      ],
      install_requires=[
          'pyrouge',
          'moverscore',
          'bert-score',
          'gin-config',
          'en_core_web_sm',
          'en_core_web_md',
          'wmd',
          'stanza',
          'transformers>=2.2.0',
          'spacy==2.2.0',
          'sacrebleu',
          'nltk',
          'scipy',
      ],
)
