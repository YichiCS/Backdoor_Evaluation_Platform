dataset=cifar10
poison_rate=0.02
cover_rate=0.04
poison_type=adp_patch
cleanser=ac


python create_clean_dataset.py -dataset $dataset

python create_poisoned_dataset.py -dataset $dataset -poison_type $poison_type -poison_rate $poison_rate -cover_rate $cover_rate

python train.py -dataset  $dataset -poison_type $poison_type -poison_rate $poison_rate -cover_rate $cover_rate

python test.py  -dataset  $dataset -poison_type $poison_type -poison_rate $poison_rate -cover_rate $cover_rate

python clean.py -dataset  $dataset -poison_type $poison_type -poison_rate $poison_rate -cover_rate $cover_rate -cleanser $cleanser

python retrain.py -dataset  $dataset -poison_type $poison_type -poison_rate $poison_rate -cover_rate $cover_rate -cleanser $cleanser

python test.py  -dataset  $dataset -poison_type $poison_type -poison_rate $poison_rate -cover_rate $cover_rate  -cleanser $cleanser