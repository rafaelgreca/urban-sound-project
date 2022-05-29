# Urban Sound Project

## How to run

```bash
conda create --name env --file environment.yml & conda activate env
```

```python
python3 main.py --seed 23 --epochs 500 --lr 0.001 --n_mfcc 40 --test_size 0.2 --dropout_rate 0.2
```