# Behavioral Differences in Mode-Switching Exploration for RL

Code for ICLR 2024 blog post track paper entitled 
[*Behavioral Differences in Mode-Switching Exploration for Reinforcement 
Learning*](https://iclr-blogposts.github.io/2024/blog/mode-switching/). 
See Section 2 of the blog post for a summary of the code.  


### Requirements

The requirements are found in the `requirements.txt` file. This code has been 
tested on Ubuntu 22.04. 

### Generating Data

There are two scripts that are used to gather the data. Details on the game 
names, environment preprocessing, and some other constants are located in 
the `common/utils.py` file.

- The first script is `data_generation/policy_generator.py`, which trains a 
  DQN agent on a specified environment. The policies are stored in the 
  location `data/<game_name>/runs`.
- The second script is `data_generation/signal_generator.py`, which evaluates 
  the trained policies. A folder named `scores` must be made under each 
  `data/<game_name>` folder, and the evaluation runs store signals in that 
  folder for each game.

### Analyzing Data

The data are analyzed in five experiments in the folders 
`experiments/exp_X` where X ranges from 1 to 5. These numbers correspond to 
the order of the experiments in the blog post. 

- The main experiments are analyzed with the script 
  `experiments/exp_X/analyzer.py` that creates data file 
  `experiments/exp_X/exp_X_data.p` which is graphed by the 
  `experiments/exp_X/grapher.py` script. 
- A toy experiment to help illustrate the results or ideas of the 
  original experiment uses the `experiments/exp_X/illustrative_analyzer.py` 
  script that creates data file `experiments/exp_X/exp_X_illustration_data.p` 
  which is graphed by the `experiments/exp_X/illustrative_grapher.py` script. 