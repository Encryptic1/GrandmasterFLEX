#!/usr/bin/env python3
import os
import chess.pgn
import numpy as np
import time
from state import State
from colorama import Fore, Back, Style, init
import termcolor
init(autoreset=True)
from termcolor import colored
class gendata():
  def get_dataset(num_samples=None):
    X,Y = [], []
    gn = 0
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    # pgn files in the data folder
    data = 'C:\\Users\\cjsli\\source\\repos\\Grandmasterflex\\Twitchchess\\TrainingGames'
    print(colored('****************_____________________GO for ' + str(num_samples) +' Examples_____________________****************','cyan'))
    go = 1
    for fn in os.listdir(data):
      if go == 0:
        break
      pgn = open(os.path.join(data, fn))
      while go == 1:
        game = chess.pgn.read_game(pgn)
        if game is None:
          print('nogame')
          break
        res = game.headers['Result']
        if res not in values:
          continue
        value = values[res]
        board = game.board()
        #print('moves')
        for i, move in enumerate(game.mainline_moves()):
          board.push(move)
          ser = State(board).serialize()
          X.append(ser)
          Y.append(value)
          #only print every 1000 games loaded
        if (gn % 1000 == 0):
          try:
            eta = time.time() - start
          except:
            eta = 0
            start = time.time()
          listlen = len(X)
          #timer over 1000 =/1000 --> /timer = seconds --> /60= minutes
          try:
            donein = (((num_samples-listlen)/1000)/eta)/60
          except:
            donein = 1.001
          #print('test')
          print("parsing game %d, got %d examples in %.3f seconds/1000 games - Done in: +- %.3f Minutes" % (gn, listlen, eta, donein))
          start = time.time()
        #gn1 = (gn += 1)
          #Plotter.plot('game', 'examples', 'dataset growth' , gn, listlen)
          #progress = listlen/num_samples
          #MainWindow.updatepbar(progress)
        if num_samples is not None and len(X) > num_samples:
          go = 0
        gn += 1
    X = np.array(X)
    Y = np.array(Y)
    #print('saving')
    totallen = str(len(X))
    namenet = "Processed/dataset_"+ str(totallen) +"_EX.npz"
    print('End of dataset reached @: ', totallen,'\n','Saved as: ',namenet)
    np.savez(namenet, X, Y)

    

