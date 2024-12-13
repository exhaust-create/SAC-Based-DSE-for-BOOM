# Description
This is the open-source code of our SAC-based micro-architecture DSE method.

# Environment
We operated this program on Windows 10 or Windows 11, python 3.8 or python 3.10. The computer memory should have at least 16 GB.

# Run
To run the program, execute the command like `python main.py --width_pred 5W_721 > ./log/log.log`, where `--width_pref 5W_721` refers to the DecodeWidth and the weight vector which is `[0.7,0.2,0.1]`.

In the log file, search for the word "projection", then you will see the projection value of the Best Found Micro-Architecture Design in each trial. The form of the message is like this:
```
XXX
```
