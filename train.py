import neat
import retro
import os
import cv2
import numpy as np
import pickle
import visualize
import send_mail
from PIL import Image as im
import multiprocessing

#Enviroment object (Allows us to run the game and gather game infomation)
env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")


def eval_genome(genome, config):
    '''
    Runs the first level of the Sonic game for each genome. It determines the fitness level for each genome.

    Parameter:
    _________
    genome: The genome
    config: The NEAT Configuration file as an object.
    '''
    genome.fitneess = 4.0
    ob = env.reset()
    inx, iny, inc = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)
    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    fitness_current = 0
    frame = 0
    xpos = 0
    done = False
    oldX = 80
    oldRings = 0
    while done is False:
        env.render()
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx,iny))
        imgarray = np.ndarray.flatten(ob)
        nnOutput = net.activate(imgarray)
        ob, rew, done, info = env.step(nnOutput)
        xpos = info['x']
        rings = info['rings']
        frame += 1
        if xpos >= 10000:
            fitness_current += 5818950
            done = True
        if(xpos > oldX):
            fitness_current += xpos - 80
        elif(xpos == oldX):
            fitness_current -= (0.5*oldX)
        else:
            fitness_current -= oldX

        if rings < oldRings:
            fitness_current -= 5000

        if info['lives'] < 3:
            fitness_current -= 25000 
            genome.fitness = fitness_current
            done = True
        
        oldX = xpos
        oldRings = rings
        if done or fitness_current < -8000 or frame == 2000:
            done = True
            print(fitness_current)
    return fitness_current


def run(config_file):
    '''
    Imports configuration file, set ups training space, runs trainings, creates the visulaizations and send the best genome and the visulizations to Karen and I

    Parameter:
    _________
    config_file: Path to config file
    '''
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate)
    
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True) 
    try:
        send_mail.SendNNData("anjolaolubusi@gmail.com")
        send_mail.SendNNData("ksuzue22@wooster.edu")
    except:
        print("Could not send email")

def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)

if __name__ == "__main__":
    main()
