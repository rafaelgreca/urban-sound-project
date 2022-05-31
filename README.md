## Urban Sound Project

Using PyTorch to create a Multi Layer Perceptron (Feed Forward Neural Network) to classify Urban Sound using the Urban Sound 8k dataset.

### Prerequsites

Make sure to store **ONLY** the 10 sub-folders and the UrbanSound8K.csv within a folder called **data**.

### How to run

```bash
conda create --name env --file environment.yml && conda activate env
```

```python
python3 main.py --seed 23 --epochs 500 --lr 0.001 --n_mfcc 40 --test_size 0.2 --dropout_rate 0.2
```