# -*- coding: utf-8 -*-
import os



if __name__ == '__main__':
    
    learning_rate=1
    for i in range(5):
        learning_rate=learning_rate*0.5
       # train 1 epoch
        print('################    train    ################')
        os.system('python3 train.py' +' --learning_rate={0}'.format(learning_rate) )

        # eval
        print('################    eval    ################')
        os.system('python3 evaluate.py' )


        # inference
        print('################    inference    ################')
        os.system('python3 run_inference.py')


