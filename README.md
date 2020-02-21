# Learning from Positive & Unlabeled Data with Arbitrary Positive Shift

**Authors**: [Zayd Hammoudeh](https://zaydh.github.io) & [Daniel Lowd](https://ix.cs.uoregon.edu/~lowd/)  
**Link**: [Arxiv](https://arxiv.org/abs/2002.10261)  
**License**: [MIT](LICENSE)  

This repository contains the source code for reproducing the results in the paper "Learning from Positive and Unlabeled Data with Arbitrary Positive Shift"

## Running the Program

To run the program, enter the `src` directory and call:

`python driver.py ConfigFile`

where `ConfigFile` is one of the `yaml` configuration files in folder [`src/configs`](src/configs). If CUDA is installed on your system, the program enables CUDA execution automatically.

### First Time Running the Program

The first time each configuration is run, the program automatically downloads any necessary dataset(s) and create any transfer learning representations automatically.  Please note that this process can be time consuming --- in particular for 20 Newsgroups where creating the ELMo-based embeddings can take several hours.

These downloaded files are stored in a folder `.data` that is in the same directory as `driver.py`.  If the program crashes while running a configuration for the first time, we recommend deleting or moving the `.data` to allow the program to redownload and reinitialize the source data.

### Results

Results are printed to the console. The tool also creates a folder named `res` in the same directory as `driver.py` where it exports results in CSV (comma separated value) format.  It includes results for all learners using metrics:

* Accuracy
* AUROC
* F1 Score

### Requirements

Our implementation was tested in Python 3.6.5.  Minimum testing was performed with 3.7.1 but [`requirements.txt`](src/requirements.txt) may need to change depending on your local Python configuration.  It uses the [PyTorch](https://pytorch.org/) neural network framework, version 1.3.1.  For full Python package dependencies, see `requirements.txt` in the `src` directory.

### Acknowledgements

This repository includes an implementation of PUc [1] that was provided by the tool's author [Tomoya Sakai](https://t-sakai-kure.github.io/).

## References

[1] Tomoya Sakai and Nobuyuki Shimizu. Covariate shift adaptation on learning from positive and unlabeled data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 4838-4845, July 2019.
