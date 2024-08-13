Assignment for course AE3540 - Bio-inspired Intelligence and Learning for Aerospace Applications

The assignment is about using a Reinforcement learning algorithm for landing the lunar lander using the OpenAI-Gymnasium library. Proximal Policy Optimization (PPO) has been used as the algorithm to make the agent learn the right actions to take to successfully land the lander.

The files in the repository are as explained:

1. functions.py: This is the Python file where all the functions required for various purposes within the project are defined.
2. user_game.py: Through this, any user can try playing the lunar lander game using their keyboard to understand what is happening and what is required to achieve a successful landing. A plot with live rewards is also generated and depending on the screen resolution one might have to move the rewards plot/game interface away from each other to properly use it.
3. Single_iteration.py: In this Python file, a single iteration with a random set of hyperparameters is performed to train the agent to obtain a reliable landing solution.
4. hyperparameter_tuning.py: This is the code where the variation of hyperparameters is performed and the plots obtained through this are included in the report.
5. best_solution.py: Depending on the hyperparameters tuning, a single iteration is performed to train the agent to obtain the best landing solution.
6. test.py: Based on the hyperparameters, the learned parameters for the neural network are saved, and using this test.py Python file, the final result can be visualized without running the entire training process.


** Note: Re-training might not produce the same results as the agent learns best on the current conditions that the environment gives it and hence replicability is not guaranteed even after many runs.
