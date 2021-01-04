### BrainNetCNN_Personality
***
#### What is this pipeline's intended use?
Nested k-fold cross validation for prediction (i.e. regression, classification) of subjects' psychometric/demographic
measures

#### What models are implemeted here?
1. PyTorch implementation of BrainNetCNN (Kawahara et al. 2016)
2. Scikit-learn implementation of SVM, MLP, and ElasticNet.

#### What training data should be used?
Square matrices. Originally, graph theoretical (e.g. correlation) matrices derived from neuroimaging data

#### In what format should the training data be?
Info about subjects should be a .csv file, including "Subject" and any confounds or outcomes as column headers.
"Subject" should list subject IDs as integers.

A matrix for one subject should should be stored as a .txt titled '{subject ID}.txt'.

#### Where do I put my data?
All data should go in the directory '/input_data/'.

A directory for matrices of the same size should be created as
 'input_data/{matrix_directory}'.

Multiple tasks completed by the same subjects should live in sub-directories of the matrix directory
(e.g. '/{matrix_directory}/task_resting_state/', '/{matrix_directory}/task_working_memory/').

If the matrix directory has a single task, its matrices should live in a sub-directory titled after the matrix directory
(e.g. '/{matrix_directory}/{matrix_directory}/')

The appropriate file path for subject info for a corresponding matrix directory is 
'/subject_info/{matrix directory}_subject_info.csv'

The matrix directory and matrix sub-directory names are called with the parser, so they should not have any spaces.

#### What is saved during training?
Trained models, the output on their highest performing epoch, and their performance results

#### How do I train a model?
Pass command line args to "parseTrainArgs.py". For help use $python parse_training_args.py -h

#### How do I automate training of multiple models?
multitrain.sh gives an example of how to automate model training with a shell script

#### How do I see the results?
Run '$python display_results/print_nested_results.py'

#### Assorted Notes
For proper handling of categorical columns in the subject info .csv,
 please add the column headers to _multiclass_variables_ in 'utils/util_args.py'

'Family_ID' column should be added in the subject info .csv, and '--families_together' called by the parser if you would
like all members of a subjects' family to be assigned to the same partition during cross-validation

If single-fold training or resumption of training after an interruption is desired, training of specific folds can be 
specified by '--start_fold' and '--end_fold'

1D network training uses nested k-fold cross validation with only a train and test set, as hyperparameter optimization
is not implemented

***

Adapted from https://github.com/nicofarr/brainnetcnnVis_pytorch, license below:

>MIT License
>
>Copyright (c) 2018 Nicolas Farrugia
>
>Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
