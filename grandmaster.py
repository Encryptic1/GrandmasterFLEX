import sys
import chess
import chess.svg
import chess.pgn
import chess.engine
from PySide2.QtCore import Slot, Qt
from PySide2.QtGui import (QBrush, QColor)
from PySide2.QtWidgets import (QApplication, QGraphicsScene, QGraphicsView, QDialog,
                               QGridLayout, QProgressBar, QStyleFactory, QPushButton, QWidget, QGraphicsObject,
                               QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QTextEdit, QLineEdit)
from PySide2.QtSvg import (QGraphicsSvgItem, QSvgRenderer, QSvgWidget)
from colorama import Fore, Back, Style, init
import termcolor
init(autoreset=True)
from termcolor import colored
from halo import Halo
import torch
import random
import time
import numpy as np
import pickle
import pandas as pd
print('Be sure to start Visdom in another terminal: visdom  & Monitor:@ http://localhost:8097')
x = input('Started Visdom server. Continue?')
import visdom
from datetime import datetime
## multi threading for GUI updates
import threading
from qt_thread_updater import get_updater
####
from utils import Plotter
#####
from generate_training_set import gendata
from state import State
#from Chess_read import CHESS_READ
#from Neural_Network import NEURAL_NETWORK
#from MCTS import MCTS
viz = visdom.Visdom()
layoutb = dict(title="GrandmasterFlex-Black", xaxis='move', yaxis='win %')
layoutw = dict(title="GrandmasterFlex-White", xaxis='move', yaxis='win %')

class Valuator(object):
    def __init__(self):
        import torch
        from train import Net
        print('Model Load')
        vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.data[0][0])

# let's write a simple chess value function
# discussing with friends how simple a minimax + value function can beat me
# i'm rated about a 1500

MAXVAL = 10000
class ClassicValuator(object):
  values = {chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0}

  def __init__(self):
    self.reset()
    self.memo = {}

  def reset(self):
    self.count = 0

  # writing a simple value function based on pieces
  # good ideas:
  # https://en.wikipedia.org/wiki/Evaluation_function#In_chess
  def __call__(self, s):
    self.count += 1
    key = s.key()
    if key not in self.memo:
      self.memo[key] = self.value(s)
    return self.memo[key]

  def value(self, s):
    b = s.board
    # game over values
    if b.is_game_over():
      if b.result() == "1-0":
        return MAXVAL
      elif b.result() == "0-1":
        return -MAXVAL
      else:
        return 0

    val = 0.0
    # piece values
    pm = s.board.piece_map()
    for x in pm:
      tval = self.values[pm[x].piece_type]
      if pm[x].color == chess.WHITE:
        val += tval
      else:
        val -= tval

    # add a number of legal moves term
    bak = b.turn
    b.turn = chess.WHITE
    val += 0.1 * b.legal_moves.count()
    b.turn = chess.BLACK
    val -= 0.1 * b.legal_moves.count()
    b.turn = bak

    return val

def explore_leaves(s, v):
        ret = []
        rndm = random.randint(2,12)
        print(colored(Style.BRIGHT + 'Thinking about your move ','magenta') + colored(str(rndm),'cyan') + colored(Style.BRIGHT + ' Out.... Probably more O.o','magenta'))
        spinner = Halo(text='SEARCHING MINDSTATE',text_color='cyan', spinner='simpleDotsScrolling',color='cyan')
        spinner.start()
        start = time.time()
        v.reset()
        bval = v(s)
        try:
            cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
            eta = time.time() - start
            spinner.stop()
            print(colored(Style.BRIGHT + "%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec",'yellow') % (bval, cval, v.count, eta, int(v.count/eta)))
            return ret
        except:
            spinner.stop()    

def computer_move(s, v):
    try:
        move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
    except:
        move = []
    if len(move) == 0:
        m1 = "game over"
        return m1
    print(colored(Style.BRIGHT + "top 3:",'green'))
    i = 0
    for i,m in enumerate(move[0:3]):
        i += 1
        mi = str(m)
        m1 = mi.split("(",)[1]
        m2 = m1.split(",",)[0]
        m= mi.split("'",)[1]
        if i == 1:
            prob = m2
            print("  ",colored(Style.DIM+ "Value increase: ",'green'),colored(Style.BRIGHT + m2,'cyan'),colored(Style.DIM + " for move ",'green'),colored(Style.BRIGHT + m,'cyan'))
        if s.board.turn == False:
            comp = colored(Back.WHITE + Fore.BLACK + Style.DIM + "Black")
        else:
            comp = colored(Back.MAGENTA + Fore.CYAN + Style.BRIGHT + "White")
    #readout = str(comp, colored(Style.BRIGHT + "moving",'magenta'), colored(Style.BRIGHT + str(move[0][1]),'yellow'))
    print(comp, colored(Style.BRIGHT + "moving",'magenta'), colored(Style.BRIGHT + str(move[0][1]),'yellow'))
    m1 = str(move[0][1])
    #game = chess.pgn.Game()
    return m1,s,prob

def computer_minimax(s, v, depth, a, b, big=False):
    if depth >= 5 or s.board.is_game_over():
        return v(s)
    # white is maximizing player
    turn = s.board.turn
    if turn == chess.WHITE:
        ret = -MAXVAL
    else:
        ret = MAXVAL
    if big:
        bret = []
        
    # can prune here with beam search
    isort = []
    for e in s.board.legal_moves:
        s.board.push(e)
        isort.append((v(s), e))
        s.board.pop()
    move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

    # beam search beyond depth 3
    if depth >= 3:
        move = move[:10]
    for e in [x[1] for x in move]:
        s.board.push(e)
        
        tval = computer_minimax(s, v, depth+1, a, b)
        s.board.pop()
        if big:
            bret.append((tval, e))
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(a, ret)
        if a >= b:
            break  # b cut-off
        else:
            ret = min(ret, tval)
            b = min(b, ret)
        if a >= b:
            break  # a cut-off
    
    if big:
        return ret, bret
    else:
        return ret

class MainWindow(QWidget):
    """
        Main UI Window
    """
    def __init__(self, parent=None):
        """
        Initialize the chess board.
        """
        super().__init__(parent)
        ## NN param set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Processing device: ',self.device)
        self.path = "best_policy.model"
        self.learning_rate   = 0.0001
        self.mcts_iteration = 100
        #self.Chess_read     = CHESS_READ()
        ##
        self.setWindowTitle("GrandmasterFLEX")
        self.resize(1280, 720)
        self.svgWidget = QSvgWidget(parent=self)
        self.svgWidget.setGeometry(10, 10, 700, 700)
        self.boardSize = min(self.svgWidget.width(), self.svgWidget.height())
        self.margin = 0.05 * self.boardSize 
        self.squareSize = (self.boardSize - 2*self.margin)/8
        self.board = chess.Board()
        self.lastmove = None
        self.pieceToMove = [None, None]
        self.drawBoard()
        self.enginePath = "./stockfish.exe"
        self.time = 0.1

        self.pbar = QProgressBar(self)
        self.pbarlabel = QLabel('Training Progress: ')
        self.maxtimelabel = QLabel("Max number of examples from ./Training Games")
        self.engineLabel = QLabel("Selfplay games#")
        self.engineButton = QPushButton("Find move")
        self.moveButton = QPushButton("Make move")
        self.undoButton = QPushButton("Undo move")
        self.testButton = QPushButton("Selfplay")
        self.gendataButton = QPushButton("Generate Dataset")
        self.trainButton = QPushButton("Train AI")
        self.newgameButton = QPushButton("New Game")
        self.undoButton.setMaximumWidth(175)
        self.pathField = QLineEdit()
        self.pathField1 = QLabel("0")
        self.examples = QLineEdit()
        self.engineResult = QLineEdit()
        self.pathField.setText("5")
        self.examples.setText('10000000')
        self.pbar.setValue(0)

        self.main_layout = QHBoxLayout()
        self.tool_layout = QGridLayout()
        ### widget design
        ##QT top row
        self.tool_layout.addWidget(self.engineLabel, 0, 0)
        self.tool_layout.addWidget(self.pathField, 0, 1)
        self.tool_layout.addWidget(self.pathField1, 0, 2) 
        self.tool_layout.addWidget(self.testButton, 0, 3)
        #second row
        self.tool_layout.addWidget(self.maxtimelabel, 1, 0)
        self.tool_layout.addWidget(self.examples, 1, 1)
        self.tool_layout.addWidget(self.gendataButton, 1, 2)
        self.tool_layout.addWidget(self.trainButton, 1, 3)
        #third row
        self.tool_layout.addWidget(self.pbarlabel, 2, 0)
        self.tool_layout.addWidget(self.pbar, 2, 1)
        #4th row
        self.tool_layout.addWidget(self.engineButton, 3, 0)
        self.tool_layout.addWidget(self.moveButton, 3, 1)
        self.tool_layout.addWidget(self.engineResult, 3, 2)
        self.tool_layout.addWidget(self.undoButton, 3, 3)
        self.tool_layout.addWidget(self.newgameButton, 3, 4)
        
        ###
        self.main_layout.addWidget(self.svgWidget, 55)
        self.main_layout.addLayout(self.tool_layout, 45)        
        self.setLayout(self.main_layout)

        self.engineButton.clicked.connect(self.find_move)
        self.moveButton.clicked.connect(self.make_move)
        self.undoButton.clicked.connect(self.undo_move)
        self.testButton.clicked.connect(self.selfplay)
        self.gendataButton.clicked.connect(self.gendata)
        self.trainButton.clicked.connect(self.trainnet)
        self.newgameButton.clicked.connect(self.newgame)
        
        #self.NN         = NEURAL_NETWORK(self.learning_rate, self.path, self.device)
        #self.mcts_tree  = MCTS(self.mcts_iteration, self.NN, self.Chess_read, self.device)
    def updarepbar(epoch1):
        self.pbar.setValue(epoch1)

    def gendata(self):
        numexamples = int(self.examples.text())
        gendata.get_dataset(numexamples)
        print(colored('Model saved','green'))

    def trainnet(self):
        with open("train.py") as f:
            code = compile(f.read(), "train.py", 'exec')
            exec(code)

    def selfplay(self):
        x = self.pathField.text()
        if x == '':
            print('no games')
            return
        else:
            self.itr = 1
            while self.itr <= int(x):
                while not self.board.is_game_over():
                    self.pathField1.setText('#games played: '+ str(self.itr))
                    go = self.make_move()
                    if go =='gameover':
                        self.post()
                        self.itr += 1
                        return
                print("GAME OVER: game: ",str(self.itr))
                self.newgame()

                
    def newgame(self):
        moves = []
        x1 = np.array([0])
        y1 = np.array([0])
        viz.line(x1,y1,update='replace',opts=layoutb,name='GrandmasterFlex - BLACK',win='GrandmasterFlex3')
        viz.line(x1,y1,update='replace',opts=layoutw,name='GrandmasterFlex - WHITE',win='GrandmasterFlex2')
        self.board.reset()
        s.board.reset()
        with open('si.pickle', 'wb') as p:
            pickle.dump(s, p)
        i = 0
        with open('rnd.pickle','wb') as rnd:
            pickle.dump(i,rnd)
        with open('g.pickle','wb')as g:
            moves = []
            pickle.dump(moves, g)

    @Slot(QWidget)
    def mousePressEvent(self, event):
        """
        Handles the left mouse click and enables moving the chess pieces by first clicking on the chess
        piece and then on the target square. 
        """
        if not self.board.is_game_over():
            if event.x() <= self.boardSize and event.y() <= self.boardSize:
                if self.margin<event.x()<self.boardSize-self.margin and self.margin<event.y()<self.boardSize-self.margin:
                    if event.buttons()==Qt.LeftButton:
                        file = int((event.x() - self.margin) / self.squareSize)
                        rank = 7 - int((event.y() - self.margin) / self.squareSize)
                        square = chess.square(file, rank)
                        piece = self.board.piece_at(square)
                        coordinates = "{}{}".format(chr(file + 97), str(rank + 1))
                        if self.pieceToMove[0] is not None:
                            move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], coordinates))
                            if move in self.board.legal_moves:
                                self.board.push(move)
                                s.board.push(move)
                            self.lastmove = str(self.pieceToMove[1]) + str(coordinates)
                            print('Last Move: ',self.lastmove)
                            piece = None
                            coordinates = None
                            with open('rnd.pickle','rb') as rnd:
                                i = pickle.load(rnd)
                                i = 1 + i
                            with open('rnd.pickle','wb') as rnd:
                                pickle.dump(i,rnd)
                            print(colored("move: " + str(i),'yellow'))
                        self.pieceToMove = [piece, coordinates]
                        self.drawBoard()
                        go = self.make_move()
                        if go == 'gameover':
                            print(colored(Style.BRIGHT + "********************* GAME IS OVER *********************",'red'))
                            self.post()
    
    def drawBoard(self):
        """
        Draw a chess board for the starting position and then redraw it for every move.
        """
        self.svgBoard = self.board._repr_svg_().encode("UTF-8")
        self.svgDrawBoard = self.svgWidget.load(self.svgBoard)  
        return self.svgDrawBoard


    def find_move(self,lastmove):
        """
        Calculate the best move according to Stockfish
        """
        #spinner = Halo(text='SEARCHING MINDSTATE',text_color='cyan', spinner='simpleDotsScrolling',color='cyan')
        #spinner.start()
        with open('si.pickle', 'rb') as f:
            s = pickle.load(f)
        try:
            times = float(self.engineTime.text())
        except:
            times = float(1)
        self.time = times
        #engine = chess.engine.SimpleEngine.popen_uci(self.enginePath)
        #result = engine.play(self.board, chess.engine.Limit(time=self.time))
        print(self.board,'\n',self.lastmove)
        #move, prob = self.mcts_tree.play(self.board, self.lastmove)
        move,s,prob = computer_move(s, v)
        self.engineResult.setText(str(move))
        #engine.quit()
        #spinner.stop()
        return move,s, prob
        


    def make_move(self):
        """
        Finds and plays the best move
        """
        try:
            with open('rnd.pickle','rb') as rnd:
                i = pickle.load(rnd)
                i = 1 + i
        except:
            i = 0
        with open('rnd.pickle','wb') as rnd:
            pickle.dump(i,rnd)
        try:
            move,s,prob = self.find_move(self.lastmove)
            print(colored('move: ' + str(i),'yellow'))
            if (i % 2) == 0:
                prob1 = np.array([float(prob)])
                i1 = np.array([(i/2)])
                #viz.line(prob1,i1,update='append',opts=layoutb,name='GrandmasterFlex - BLACK',win='GrandmasterFlex3')
                plotter.plot('Round', 'Prob', 'Win Probability - Black' , i1, prob)
            else:
                prob1 = np.array([float(prob)])
                i1 = np.array([(i/2)])
                #viz.line(prob1,i1,update='append',opts=layoutw,name='GrandmasterFlex - WHITE',win='GrandmasterFlex2')
                plotter.plot('Round', 'Prob', 'Win Probability - White' , i1, prob)
            #print(move)
            nf3 = chess.Move.from_uci(move)
            #print('uci eq: ',nf3)
            #x = input('waiting')
            with open('g.pickle','rb')as g:
                    moves = pickle.load(g)
                    moves.append(move)
            with open('g.pickle','wb')as g:
                pickle.dump(moves, g)
                print(colored(moves,'cyan'))

            self.board.push(nf3)
            s.board.push(nf3)
            self.drawBoard()
            time.sleep(0.2)
            QApplication.processEvents()
            self.lastmove = move
            with open('si.pickle', 'wb') as p:
                pickle.dump(s, p)
        except:
            print(colored(Style.BRIGHT + "********************* GAME IS OVER *********************",'red'))
            #self.post()
            return 'gameover'
        

    def undo_move(self):
        """
        Undo the last move played on the board
        """
        self.board.pop()
        s = self
        self.drawBoard()

    def post(self):
        game = chess.pgn.Game()
        game.headers["Event"] = "GMFLEX"
        game.headers["Site"] = "local"
        game.headers["Date"] = datetime.now()
        game.headers["White"] = "White"
        game.headers["Black"] = "Black"
        with open('g.pickle','rb')as g:
            moves = pickle.load(g)
            i = 0
            for move in moves:
                i += 1
                if i == 1:
                    print(move)
                    node = game.add_variation(chess.Move.from_uci(move))
                else:
                    node = node.add_variation(chess.Move.from_uci(move))
        with open('si.pickle', 'rb') as f:
            si = pickle.load(f)
            b = si.board
            # game over values
            if b.result() == "1-0":
                game.headers["Result"] = "1-0"
                winner = "WHITE"
            elif b.result() == "0-1":
                game.headers["Result"] = "0-1"
                winner = "BLACK"
            else: 
                winner = "BLACK"
            win = open('winner.txt', 'a')
            stamp = str(datetime.now()).replace(":","_")
            log = open('TrainingGames/'+ stamp +".pgn", 'w')
            print(game, file=log, end="\n\n")
            try:
                sgame = "Final Board:\n" + str(game)
            except:
                print(colored(Style.BRIGHT + "Save Game Fail",'red'))
            win.write(winner + ":\n" + sgame)
            if winner == "BLACK":
                res1 = 0
                res2 = 1
            else:
                res1 = 1
                res2 = 0
        with open('w.pickle','rb')as w:
                wins = pickle.load(w)
        win1 = pd.DataFrame({'W':[res1],'B':[res2],'Winner':[winner]})
        pd.DataFrame.append(wins,win1)
        with open('w.pickle','wb')as w:
            pickle.dump(wins,w)
        with open('rnd.pickle','wb') as rnd:
            pickle.dump(i,rnd)
        with open('g.pickle','wb')as g:
            moves = []
            pickle.dump(moves, g)
    def update(self):
        """Update the gui with the current value."""
                  
        self.drawBoard()


if __name__ == "__main__":
    global plotter
    plotter = Plotter(env_name='GrandmasterFLEX')
    app = QApplication(sys.argv)
    gui = MainWindow()
    # chess board and "engine"
    s = State()
    wins=pd.DataFrame({'W':[1,0],'B':[0,1],'Winner':['White','Black']})
    with open('w.pickle','wb')as w:
        pickle.dump(wins,w)
    with open('si.pickle', 'wb') as p:
        pickle.dump(s, p)
    i = 0
    with open('rnd.pickle','wb') as rnd:
        pickle.dump(i,rnd)
    with open('g.pickle','wb')as g:
        moves = []
        pickle.dump(moves, g)
    #v = Valuator()
    v = ClassicValuator()
    ## gui update thread
    gui.show()
    sys.exit(app.exec_())