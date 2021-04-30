# MLSonic
This project is designed to test a machine learning algorithm against the sonic games. This algorithm will try and complete the level.

## Setup
First, make sure you are using Python 3.8. Secondly, have a virtual enviroment. Then, proceed to install the neccessary packages by opening the cloned directory in a console and typing:
```
pip install -r requirements.txt
```
Once all the packages has been installed, copy the rom.md file to ```[NAME OF VIRTUAL ENV]/lib/python3.8/site-packages/retro/data/stable/SonicTheHedgehog-Genesis```

## Running
To run the NEAT training, you can type: 
```
python train.py
```
To see the best genome complete the level, you can type:
```
python test-genome.py
```


## Authors
- [Anjolaoluwa Olubusi](https://github.com/anjolaolubusi)
- [Karen Suzue](https://github.com/karensuzue)

