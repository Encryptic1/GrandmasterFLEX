# MIB Chess

<img width=600px src="https://raw.githubusercontent.com/Encryptic1/twitchchess/master/m1.PNG" />
Agents K and J will play selves
Agent K will play white = human

modded from git GEOHOT/twitchchess
- Added more HTML feeback and colorized console, added more exceptions and feedback in waiting.
- trained on grandmaster games to .017% loss rate saved to new net
- built in replies from index.html and a game posting function
- built log for saving to TrainingGames/ and post()
	- Randomized first move in selfplay for training sets. commented out original
- debugged UI a bit

<img width=600px src="https://raw.githubusercontent.com/Encryptic1/twitchchess/master/m2.PNG" />

Its still a bit buggy interface but: 
- moving a peice off board will reset the peice to original position
- to manually post a game to pgn use concede and post button below board
- must reset board with new game befor next human game
- selfplay will do its thing for you

<img width=600px src="https://raw.githubusercontent.com/Encryptic1/twitchchess/master/hm1.PNG" />

human play will give both shell and html feedack
- while the program is exploing leaves any input will cause a break
- must wait until your turn to ff 
- new game resets board without posting game

*************************************************************
MAKE SURE: 
you need to make a folder called TrainingGames in the root directory for game dumps and for generating training set.
*************************************************************

	TODO:
	- selftrain set
	-  add policy condition(make utility move) for tie values, in lategame the bot plays safe. wheres the aggro?

    CNN config
    - added LR to optimizer and batch sheduler.
    - learning rate spiking due to adam batching need to control and reduce learning rate on the larger than 5million example sets
    -25m example set failed to find global min. gradent decent is a possability even though adam is more efficient

*************************************************************
USAGE: 
```
git clone repository
```
Install dependancies:
```
pip install -r requirements.txt
```
Running the application for (generating dataset/playing/using model/UI)
```
cd dir*
python grandmaster.py
```
Running a training session (~ 3mim/ epoch @10M examples. Device: CUDA RTX 3090)
```
python train.py
```
be sure to use the mane of the training set you either generated or want to use in line 26 train.py the name will be associated with the number of examples contained in the .npz
```
dat = np.load("processed/dataset_**********_EX.npz")
```
you will be prompted to enter the number of epochs you want to use in a non decimal int() dtype
*************************************************************