import neat
import retro
import os
import cv2
import numpy as np

env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act2")
def eval_genomes(genomes, config):
    current_max_fitness = 0
    for genome_id, genome in genomes:
        genome.gitneess = 4.0
        neat.nn.FeedForwardNetwork.create(genome, config)
        ob = env.reset()
        ac = env.action_space.sample()
        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        done = False
        env.render()
        frame += 1
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx,iny))
        imgarray = []
        imgarray = np.ndarray.flatten(ob)
        nnOutput = net.activate(imgarray)
        ob, rew, done, info = env.step(nnOutput)
        xpos = info['x']
        if xpos >= 10000:
            fitness_current += 10000
            done = True
        fitness_current += rew
        if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
        else:
                counter += 1
        if done or counter == 250:
            done = True
            print(genome_id, fitness_current)
                
        genome.fitness = fitness_current


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)

if __name__ == "__main__":
    main()
