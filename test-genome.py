import retro
import numpy as np
import neat
import cv2
import pickle
import os
env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")
ob = env.reset()
inx, iny, inc = env.observation_space.shape
inx = int(inx/8)
iny = int(iny/8)

done = False

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
winner = pickle.load(open("winner.pkl", "rb"))
winner_net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)
"""while done is False:
    env.render()
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))
    imgarray = np.ndarray.flatten(ob)
    nnOutput = winner_net.activate(imgarray)
    ob, rew, done, info = env.step(nnOutput)
"""
f = open("demofile2.txt", "a")
f.write(str(winner))
f.close()
