## Below are examples for automated training. Fill in accordingly your own data labels and desired training parameters
models=('BNCNN' 'SVM' 'FC90' 'ElasticNet')
transformations=('untransformed' 'tangent')
matrix_directory=('cfHCP900_FLS_GM')
tasks=('rest1' 'working_memory')
outcome_names=('PSQI_Score' 'Handedness')

# single input task, single outcome, combinatorial
for transform in "${transformations[@]}"; do
  for model in "${models[@]}"; do
    for outcome in "${outcome_names[@]}"; do
      for task in "${tasks[@]}"; do
        python parseTrainArgs.py --transformations "$transform" -md "${matrix_directory[0]}" -on "$outcome" -mo "$model" -t "$task" --scale_features --verbose
      done
    done
  done
done

# multiple input tasks, multiple outcome_names, multiple confounds
confound_names=('Gender' 'Age_in_Yrs')
python parseTrainArgs.py -cn "${confound_names[@]}" --deconfound_flavor X1Y1 --scale_confounds -on "${outcome_names[@]}" -mo "${models[0]}" -md "${matrix_directory[0]}" --transformations "$transform" --verbose
