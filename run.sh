#!/bin/bash
for model in natural adv_trained;
    do 
    python fetch_model.py $model
done

for model in naturally_trained adv_trained; 
    do
    for b in True False; 
	do
	python eval_run.py $model $b
    done
done
