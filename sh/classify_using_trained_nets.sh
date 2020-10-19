#!/bin/bash

#### For classifying using trained networks loaded from .h5 files

proj_dir=$HOME/SNN_practice
h5_dir=$proj_dir/h5

start_time=`date +%Y-%m-%dT%H:%M:%S`

echo `date +%Y-%m-%dT%H:%M:%S` INFO: Starting 0.05 classification...
/usr/bin/python3 $proj_dir/load_training_results.py --file_path_name $h5_dir/thirty_for_thirty_higher_conn_prob_1_p5.h5 --pres_duration 100 --num_pres_per_stim 100

for i in {1..9}
do
    echo `date +%Y-%m-%dT%H:%M:%S` INFO: Starting 0."$i" classification...
    /usr/bin/python3 $proj_dir/load_training_results.py --file_path_name $h5_dir/thirty_for_thirty_higher_conn_prob_1_"$i".h5 --pres_duration 100 --num_pres_per_stim 100
done

for i in {-5..5}
do
    echo `date +%Y-%m-%dT%H:%M:%S` INFO: Starting "$i" lat weight adjustment classification...
    /usr/bin/python3 $proj_dir/load_training_results.py --file_path_name $h5_dir/thirty_for_thirty_higher_conn_prob_1_1.h5 --pres_duration 100 --num_pres_per_stim 100 --lat_weight_adjustment $i
done

echo `date +%Y-%m-%dT%H:%M:%S` INFO: Done.

end_time=`date +%Y-%m-%dT%H:%M:%S`
echo "Start time: $start_time"
echo "End time: $end_time"
