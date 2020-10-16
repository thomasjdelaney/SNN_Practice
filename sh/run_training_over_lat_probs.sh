#!/bin/bash

##### For training many networks with different values for the lateral connection probability.
##### Can consider feed_forward_connection_prob:lateral_connection_prob as the E/I ratio (I suppose)

proj_dir=$HOME/SNN_practice
h5_dir=$proj_dir/h5

start_time=`date +%Y-%m-%dT%H:%M:%S`

echo `date +%Y-%m-%dT%H:%M:%S` INFO: Starting 0.01 run
echo /usr/bin/python3 $proj_dir/imitation_v1_net.py --file_path_name $h5_dir/thirty_for_thirty_higher_conn_prob_1_p1.h5 --num_target 30 --num_source 30 --source_rates_params 200 0.1 10 0.1 10 0.1 200 0.1 --duration 100000 --use_stdp --record_source_spikes --numpy_seed 1798 --conn_type fixed_prob --feed_forward_connection_prob 0.1 --lateral_connection_prob 0.01 --w_max 0.1

echo `date +%Y-%m-%dT%H:%M:%S` INFO: Starting 0.05 run
/usr/bin/python3 $proj_dir/imitation_v1_net.py --file_path_name $h5_dir/thirty_for_thirty_higher_conn_prob_1_p5.h5 --num_target 30 --num_source 30 --source_rates_params 200 0.1 10 0.1 10 0.1 200 0.1 --duration 100000 --use_stdp --record_source_spikes --numpy_seed 1798 --conn_type fixed_prob --feed_forward_connection_prob 0.1 --lateral_connection_prob 0.05 --w_max 0.1

for i in {1..9}
do
    echo `date +%Y-%m-%dT%H:%M:%S` INFO: Starting 0.$i run
    /usr/bin/python3 $proj_dir/imitation_v1_net.py --file_path_name $h5_dir/thirty_for_thirty_higher_conn_prob_1_"$i".h5 --num_target 30 --num_source 30 --source_rates_params 200 0.1 10 0.1 10 0.1 200 0.1 --duration 100000 --use_stdp --record_source_spikes --numpy_seed 1798 --conn_type fixed_prob --feed_forward_connection_prob 0.1 --lateral_connection_prob 0."$i" --w_max 0.1
done
echo `date +%Y-%m-%dT%H:%M:%S` INFO: Done.

end_time=`date +%Y-%m-%dT%H:%M:%S`
echo "Start time: $start_time"
echo "End time: $end_time"
