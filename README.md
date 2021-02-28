# Create virtual env
```
python3 -m venv env
```

# Activate virtual env
```
source env/bin/activate
```

# Install required modules
```
pip3 install -r requirements.txt
```

# How to generate training data?
```
python3 generate_training_data.py --symbol MSFT --out train_data.txt 
```

# How to train?
```
python3 train.py --mode train --filepath train_data.txt
```

