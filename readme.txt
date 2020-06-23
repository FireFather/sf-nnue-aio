I released a new binary set "stockfish-nnue-2020-05-30" for training data generation and training.
https://github.com/nodchip/Stockfish/releases/tag/stockfish-nnue-2020-05-30
Please get it before trying the below.

Training in Stockfish+NNUE consists of two phases, "training data generation phase" and "training phase".

In the training data generation phase, we will create training data with the "gensfen" command.
In the first iteration, we will create training data with the original Stockfish evaluation function.
This can be done with "stockfish.nnue-gen-sfen-from-original-eval.exe" in "stockfish-nnue-2020-05-30".
The command will be like:

uci
setoption name Hash value 32768  <- This value must be lower than the total memory size of your PC.
setoption name Threads value 8  <- This value must be equal to or lower than the number of the logical CPU cores of your PC.
isready
gensfen depth 8 loop 10000000 output_file_name trainingdata\generated_kifu.bin
quit

Before creating the training data, please make a folder for the training data. 
In the command above, the name of the folder is "trainingdata".
The traning data generation takes a long time.  Please be patient.
For detail options of the "gensfen" command, please refer learn/learner.cpp.<- In the source code (src\learn\learner.cpp)

We also need validation data so that we measure if the training goes well. 
The command will be like:

uci
setoption name Hash value 32768
setoption name Threads value 8
isready
gensfen depth 8 loop 1000000 output_file_name validationdata\generated_kifu.bin
quit

Before creating the validation data, please make a folder for the validation data.  
In the command above, the name of the folder is "validationdata".

In the training phase, we will train the NN evalution function with the "learn" command.  Please use "stockfish.nnue-learn-use-blas.k-p_256x2-32-32.exe" for the "learn" command.
In the first iteration, we need to initialize the NN parameters with random values, and learn from learning data.
Setting the SkipLoadingEval option will initialize the NN with random parameters.  The command will be like:

uci
setoption name SkipLoadingEval value true
setoption name Threads value 8
isready
learn targetdir trainingdata loop 100 batchsize 1000000 eta 1.0 lambda 0.5 eval_limit 32000 nn_batch_size 1000 newbob_decay 0.5 eval_save_interval 10000000 loss_output_interval 1000000 mirror_percentage 50 validation_set_file_name validationdata\generated_kifu.bin
quit

Please make sure that the "test_cross_entropy" in the progress messages will be decreased.
If it is not decreased, the training will fail.  In that case, please adjust "eta", "nn_batch_size", or other parameters.
If test_cross_entropy is decreased enough, the traning will befinished.
Congrats!
If you want to save the trained NN parameter files into a specific folder, please set "EvalSaveDir" option.

We could repeat the "training data generation phase" and "training phase" again and again with the output NN evaluation functions in the previous iteration.
This is a kind of reinforcement learning.
After the first iteration, please use "stockfish.nnue-learn-use-blas.k-p_256x2-32-32.exe" to generate training data so that we use the output NN parameters in the previous iteration.
Also, please set "SkipLoadingEval" to false in the training phase so that the trainer loads the NN parameters in the previous iteration.

We also could change the network architecture.
The network architecuture in "stockfish-nnue-2020-05-30" is "k-p_256x2-32-32".
"k-p" means the input feature.
"k" means "king", the one-shot encoded position of a king.
"p" means "peace", the one-shot encoded position and type of a piece other than king.
"256x2-32-32" means the number of the channels in each hidden layer.
The number of the channels in the first hidden layer is "256x2".
The number of the channels in the second and the third is "32".

The standard network architecture in computer shogi is "halfkp_256x2-32-32".
"halfkp" means the direct product of "k" and "p" for each color.
If we use "halfkp_256x2-32-32", we could need more training data because the number of the network paramters is much larger than "k-p_256x2-32-32".
We could need 300,000,000 traning data for each iteration.
