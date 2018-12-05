# CSE571-Group-Project
Group Project for CSE571, Artificial Intelligence. The topic we chose is Eligibility traces by implementing Sarsa(l) with linear function approximation

You can use the command: python gridworld.py -p SarsaLambdaAgent -x 2000 -n 2010 -l smallGrid -y=.8
to run the SARSA(l) algorithm with lambda = .8

x is the number of training sessions, n is the total number of sessions, l is the map, y is the value of lambda.

You can substitute smallGrid with bookGrid, mazeGrid, customGrid, and largeMazeGrid for viewing the algorithm in other environments.

For more environments, there is pacman.py, which you can subsitute for gridworld.py, which has the environments: 

