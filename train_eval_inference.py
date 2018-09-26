# -*- coding: utf-8 -*-
import os
import argparse


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()

if __name__ == '__main__':

    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)
    
    learning_rate=FLAGS.learning_rate
    number_of_steps=3750
    print(os.getcwd())
    for i in range(40):
       # train 1 epoch
        print('################    train    ################')
        print(os.getcwd)
        os.system('python3 ./train.py' +' --learning_rate={0}'.format(learning_rate)+' --number_of_steps={0}'.format(number_of_steps) )

        # eval
        print('################    eval    ################')
        os.system('python3 ./evaluate.py' )


        # inference
        print('################    inference    ################')
        os.system('python3 ./run_inference.py')

        learning_rate=learning_rate*0.75
        number_of_steps=number_of_steps*(i+2)


