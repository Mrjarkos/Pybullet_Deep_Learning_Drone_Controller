# Pybullet_Deep_Learning_Drone_Controller

This project facilitates the data extraction, training process and test for neural controllers in Pybullet. Incluiding neural network models, datasets for train neural controllers and files which allows to characterize the PID controller provided by Pybullet based on the response in terms of the control evaluation parameters. Further this project allows you the creation and record of a several kind of trajectories sets, create individual trajectories with specified parameters, train neural networks with different architectures and test the performance of the original controller versus the trained neural controllers applying different aerodynamic effects, which helps to get a more realistic perspective about the obtained controller.

## Instalation steps and requirements:

1. For implement this project install the versions of the following packages: 

      #### Recommended Versions:
      - python 3.8.10
      - tensorflow-gpu 2.5.0
      - keras 2.5.0
      - pybullet 3.1.6
      - numpy 1.19.5
      - stable_baselines3 1.5.0
      - gym 0.21.0
      - cycler 0.11.0
      - matplotlib 3.5.1
      - pillow 9.0.1  



2. Download the project in .zip format.
3. Run each script according to the explanations that you can find in the below sections.  

#### Note 1:
Tensorflow 2.5.0 is incompatible with numpy versions that are above to 1.19.5 version and Pybullet versions that are above to 3.1.7       are incompatible with numpy versions below to 1.20.0 version.

#### Note 2:
If you want to run the project with a machine that doesn't has gpu, set the parameter **gui** in each script as False. 

## Applications And File Descriptions:

In this section we show uses cases and funcionalities of the files within this repository. 

###  00_hello_world.py

This script runs a 50 seconds simulation, with the controller DSLPID provided by Pybullet. Where the drone realizes a hover and we confirm that the project was installed correctly. When the simulation is finished it should be displayed a graphic with the states of the drone and a three dimension graphic that describes the trajectory, as follows:

<p align = center>
<img src="https://github.com/UrielCarrero/Tesis-Matlab/blob/main/hello_world.png" width="60%" />
</p>

<p align = center>
<img src="https://github.com/UrielCarrero/Tesis-Matlab/blob/main/states_hello_world.png" width="80%" />
</p>

###  00_characterize_controller.py

This file allows to evaluates the transitory state response for the controller DSLPID provided by Pybullet for an specific axis. Besides measures the step response in terms of settling time, rise time, overshooot, steady state error and control effort by ITSE function. These meaures are printed out in the console at the end of the simultaion. The magnitude of the signal might be specified with the variable **params**, the trajectory type might be specified with the variable **trajectories**, the axis might be defined in the variable **ax** and the duration of the simulation might be defined with the parameter **duration_sec**.

The axis where it's possible to perform the test are:
- 'x' -> possition in x axis. 
- 'y' -> possition in y axis. 
- 'z' -> possition in z axis.
- 'vx' -> speed in x axis. 
- 'vy' -> speed in y axis.
- 'vz' -> speed in z axis.
- 'r' -> position in yaw axis.
- 'wr' -> speed in yaw axis.

The possible kind of trajectories to test are showed up on the explanation of the trajectories.py file.

###  01_dataset_gen.py

This file allows to simulate and record a set of trajectories automatically by defining some few parameters. For execute this script is necessary to define parameters for the disturbances to perform: the axes where the drone will suffer disturbances might be defined with the variable **DIST_STATES** (the list of axes explained above aplies for this variable too), the magnitude of the disturbances with the list **D_FACTOR**, the probability to suffer a disturbance with the list **D_PROB** and the time where the first disturbance will occur, which might be defined witht the variable **DIST_TIME**. Also it's necessary to define the number of kinds of trajectories to get in the dataset with the variable **N_UNIQUE_TRAJ**, the numer of trajectories per kind of trajectory with the variable **N_SAMPLES**, the name of the folder where the data will be save with the variable **path** and the duration of the simulation with the parameter **duration_sec**. The saved data will be saved in format .csv and you can find the specified folder whitin the folder "files", within "logs".

###  01_dataset_gen_manual.py

This file allows to simulate and save the states of an specified trajectory by manual. For execute this script is necesary to define the parameters for disturbances, the variables **path** and **duration_sec** described previously. It's necessary to define the number of trajectory to save with the variable j, the axes where the trajectories will be performed with the list **axis** and the kind of trajectory per axis with the list **trajectories**, taking in account the position where was located each axis in the list **axis** (the possible values to assign in these both list are the same showed in the description of the file 00_characterize_controller.py)

###  02_replay_trajectory.py

This file allows to plot the states of random trajectories within datasets folders from the path "logs/Datasets/" (in this path you can find the default datasets provided from this repository). For execute this script only is necessary to define the name of the folder which contains the dataset to read, it might be specified with the variable **Dataset_name**. Furthermore if you want to read an specific trajectory you can break the main for loop and assign the name of the file that you want to read to the variable **filename**.

###  03_ANN_vs_Control_lemniscate.py

This file tests the response for lemiscate trajectory along the x and y axis, with a ramp trajectory along z axis, for the original controller provided by Pybullet and the neural controller. For execute this script is necesary to define the parameters for disturbances, the variables **path** and **duration_sec** described above. Also is necessary to define the quantity of previous states to ingress to the neural controller throught the **window** variable, the path and the name of dataset folder with the variables **root** and **dataset** respectibly, the path of the pre-trained neural network model throught the variable **model_path**, the path of the "data_description" file (generated while the training is ran, which contains the data analisys from an specific dataset used to normalize the inputs) throught the variable **norm_data_path**, the list of inbound states to the neural network that are located within the list **states_list**. Also it's necessary to  assign a value to **feedback** and **flat** variables according to what's described within the script for the input dimentions.

The possible values to locate, within the list **states_list** are:
- Position states: 'x', 'y','z','p','q','r'.
- Speed states: 'vx','vy','vz','wp','wq','wr'.
- Aceleration states: 'ax','ay','az','ap','aq',ar'.
- Set-point signals: 'ux','uy','uz','ur'.

###  03_ANN_vs_Control_downwash.py - 03_ANN_vs_Control_ground.py - 03_ANN_vs_Control_drag.py  

These scripts allows you to compare the performance of the neural controller versus the controller provided by Pybullet under the downwash, ground and drag aerodinamic effects respectibly. To run this script is necessary to define the same parameters explained preciously for "03_ANN_vs_Control_lemniscate.py" script.

<p float="left">
  <img src="https://github.com/UrielCarrero/Tesis-Matlab/blob/main/Drag.png" width="31%" />  
  <img src="https://github.com/UrielCarrero/Tesis-Matlab/blob/main/Downwash.png" width="30.4%" /> 
  <img src="https://github.com/UrielCarrero/Tesis-Matlab/blob/main/Ground.png" width="36%" /> 
</p>  

### 03_Characterize_ANN_Control.py

This script facilitates to evaluate the transitory response of neural controller and the controller provided by Pybullet under defined trajectories along different defined axes, located within the **trajectories** and **axis** lists. Further to run this script is necessary to define the same parameters explained preciously for "03_ANN_vs_Control_lemniscate.py" script.

### 03_ANN_vs_LSTM_vs_Control.py

This script like the previous one, allows to evaluate the transitory response with defined trajectories and axis. But you can compare many pre-trained controller models and the controller provided by Pybullet. It's only necessary to define the list of paths where are the models with the dictionary **model_path** with the respectibly name, the list of colors for plot the trajectory of each controller with the dictionary **COLOR_CTRLS** and the rest variants explained previously for "03_ANN_vs_Control_lemniscate.py" script.   

### DronePySim.py

This script contains 2 defined classes that facilitates to perform the simulation with Pybullet and are the basis of the above scripts.

The first class is **Drone**. Where are defined all the methods and propierties of the drone which uses the original controller in Pybullet. Those propierties are: Drone's model, name, initial linear position, initial angular position, controller and control frecuency. The methods assigns the desired controller, assigns the specific PID constants, calculates the control signal and updates the drone's states.  

The second class is **PybulletSimDrone**.Where are defined all the methods and propierties that the simulation environment needs to runs in Pybullet. Those propierties are: quantity of drones, physics engine, the name of the file where the data will be save, the parameters for perform the disturbances, the simulation frequency, the control frequency, duration time of the simulation, the time between each step simulation and flags for indicate whether to use the pybullet gui, record video, plot the simulation results, use steps in the physics engine and add obstacles within the envronment. The methods assigns the drones to the environment, creates and sets up the environment, chooses, inicializates and modifies the set-point signals, performs a simulation step, sets up the logger, performs the simulation and saves/returns the state data.

### NNDrone.py

This file contains the neural network controller class and one function that allows to create and return the neural controller. This by specifying the model path, the path with the "data_description" file, the kind of neural network to implement, the inbound states to the neural controller, the list of outputs from the neural network and the quantity of previous states to ingress to the neural network (window) and whether you want a two or a three dimention input to the neural network. 

The class **NNDrone** contains methods and propierties for implement a drone with a neural controller within the Pybullet environment. Those propierties (besides the parameters explained in the previous paragrahp) are the initial linear position, the initial angular position, the model of the neural controller and the drone´s name. The methods updates the drone states, applies the normalization to the states, normalizes in reverse the outputs, performs all the pre-processing process and calculates the prediction from the model.

### trajectories.py

This script contains all the functions that allows to create and perform the desired trajectories. These trajectories are: 

- 'step'
- 'pulse'
- 'ramp'
- 'square'
- 'sin'
- 'cos'
- 'noise'
- 'sawtooth_sweep'
- 'triangular_sweep'
- 'chirp'
- 'chirp_amplin'
- 'big_step_ret0'
- 'big_step_notret0'
- 'step_notret0'
- 'ramp_step_notret0'
- 'step_ret0'
- 'random_step'
- 'stopped'

### Model_Training_LSTM.ipynb

With this script are trained the neural controller models, where if there is a gpu, it's set up for train. Subsequently the dataset is read and the data is normalized. Then the data is divided in 3 bunchs, one for train, other for validation and the last one for test. Later is defined a class for a tensorflow  data generator, which will ingress the data to the neural network with the shape required by tensorflow and facilitates the use of less RAM memory by generating the samples while the training is performed. Then we create 3 generators for each bunch that divides the data. Later we define the parameters for training like the callbacks, the optimizer, the learning rate, the metrics and the neural network arquitecture. Finally we compile the model and start the training where we especify the generators to use for training and validation, the steps per epoch and the quantity of epochs.

For evaluate the model we use the test generator and evaluate the behavior of our model for all the samples within the test bunch. Further we take a batch from the generator and we compare the prediction for this batch versus the target outputs plotting both in one figure.  






















