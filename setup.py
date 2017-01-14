from distutils.core import setup
setup(
  name = 'DoubleML',
  packages = ['DoubleML'], # 
  version = '0.1',
  description = 'The DoubleML package is intended to be used to implement the estimation procedure developed in "Double Machine  Learning for Treatment and Causal Parameters" by Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey.',
  author = 'Kramer Quist',
  author_email = 'kramer.quist@gmail.com',
  url = 'https://github.com/kquist/DoubleML-Python',
  download_url = 'https://github.com/kquist/DoubleML-Python/0.1', 
  keywords = ['Double machine learning', 'Neyman machine learning', 'orthogonalization', 'cross-fit machine learning', 'de-biased machine learning']
  classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Economists and Statisticians',

    # Pick your license as you wish (should match "license" above)
     'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3.5',
],
)
