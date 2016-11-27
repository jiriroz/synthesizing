#/bin/bash

# Take in an unit number
#if [ "$#" -ne "1" ]; then
#  echo "Provide 1 output unit number e.g. 945 for bell pepper."
#  exit 1
#fi

# Get label for each unit
path_labels="misc/synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

opt_layer="fc6"
act_layer="fc8"
xy=0

# Net
#net_weights="nets/googlenet/bvlc_googlenet.caffemodel"
#net_definition="nets/googlenet/bvlc_googlenet_updated.prototxt"

# Hyperparam settings for visualizing GoogLeNet
# Note that the learnign rate is different from that for AlexNet
iters="400"
#weights="985"
weights="99"
#rates="1.0"
rates="8.0"
end_lr=1e-10

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt
init_file="None"
#init_file="data/tabby.jpg"
mode="unit"

obj=$1

# Debug
debug=0
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

# Output dir
output_dir="output"
#rm -rf ${output_dir}
mkdir -p ${output_dir}/${act_layer}

list_files=""
for seed in {0..0}; do

for n_iters in ${iters}; do
    for w in ${weights}; do
        for lr in ${rates}; do

            L2="0.${w}"
            # Optimize images maximizing fc8 unit
            python ./act_max_dev.py \
                --act_layer ${act_layer} \
                --opt_layer ${opt_layer} \
                --xy ${xy} \
                --n_iters ${n_iters} \
                --start_lr ${lr} \
                --end_lr ${end_lr} \
                --lambd ${L2} \
                --seed ${seed} \
                --clip ${clip} \
                --bound ${bound_file} \
                --output_dir ${output_dir} \
                --init_file ${init_file} \
                --obj ${obj} \
                --mode ${mode}
    
            done
        done
    done
done
