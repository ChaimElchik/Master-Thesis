# Master-Thesis

This is the working repository for the Master Thesis

Overleaf link: https://www.overleaf.com/read/pkszxhzdwvjd#080ba4

FinishedNotebooks contains all notebooks used with explanations 

FinishedPythonFiles contains all notebooks converted to executable Python code with explanations

data.yaml file is necessary for running files

## Model Deployment and Training Using Python Files

* To Train and Deploy the Model the MOT dataset is required with the MOT files, Video files and Stereo Camera Parameter files in the respective folders mots and vids. The stereo Camera Parameters need to be in the same location as the Python Files. 

* By executing the PipeLine.py file the model gets trained and the best-chosen model 'det_best_bgr29.pt' gets deployed. To overrun the model selection a new model can be loaded up by editing the code.

* To visualize the results the Vizualize.py file needs to be run, it uses the df_to_analyze.csv file to visualize the results. by adding your own file this can be changed or by adjusting the code in the python file.

## Model Evaluation Using Python Files

* To evaluate the models performance the Evaluate_Model_Performance.py needs to be run. It requires the PipeLine.py file to first be fully run as the output files from the PipeLine.py file are required. The final evaluation output is saved into the Full_table_all_matches_for_dupes.csv file and AVG_table_all_matches_for_dupes.csv files. 



