# Activate virtual env
```
source env/bin/activate
```

# How to generate training data?
```
python generate_training_data.py --symbol MSFT --out train_data.txt 
```

# How to train?
```
python train.py --mode train --filepath train_data.txt
```

