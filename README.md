# Description
This is the open-source code of our SAC-based micro-architecture DSE method.

# Environment
We operated this program on Windows 10 or Windows 11, python 3.8 or python 3.10. The computer memory should have at least 16 GB.

# Run
To run the program, enter the command like `python main.py --width_pred 5W_721 > ./log/log_5W_721.log`, where `--width_pref 5W_721` refers to the DecodeWidth and the weight vector which is `[0.7,0.2,0.1]`. This command will autonomously search for the config files named _config\_ppo\_5W\_721.yml_ and _config\_sac\_5W\_721.yml_ to fulfill the hyperparameters, then execute all DSE methods.

In the log file, search for the word "projection", then you will see the projection value of the "best-found micro-architecture design" in each trial. The form of the message is like this:
```
best_design: [[1 1 1 3 3 2 3 3 4]], best_design_ppa: [[9.5114400e-01 2.6484000e-02 1.4600115e+06]], projection: [0.03823456]
```
If the message is like this:
```
No design is in the constraint! All `proj_with_punishment` < 0.
best_design: [1. 1. 1. 2. 3. 2. 1. 2. 4.], best_design_ppa: [[9.4688803e-01 2.6484000e-02 1.4524475e+06]], projection: -0.02232724337502704
```
it means the method could not search for a micro-architecture design that is within the constraint.

# Modify Hyperparameters, DecodeWidth, and Preference
The configuration files are in the path `./config/`. Please note that the file called _config\_ppo\_xxx_ and the file called _config\_sac\_xxx_ should be modified simultaneously.
