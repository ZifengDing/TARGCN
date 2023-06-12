# Preprocessing
If you want you run the demo on ICEWS14, ICEWS05-15 or GDELT, go to `'dataset/${DATASET}'`, and run:
`python preprocess.py`
Then you can train on corresponding datasets.

# Training
To run the training demo, please run:
`python main.py --dataset ${DATASET}`

# Testing
To test a trained model, please run:
`python main.py --dataset ${DATASET} --test --resume --name ${CHECKPOINT_NAME}`

# Generalization to unseen timestamps
Please go to `'dataset/icews14_unseen'` and run:
`python preprocess_extrapolate.py`
Then go to the root directory and run:
`python main.py --dataset icews14_unseen`

# Generalization to irregular timestamped data
Please run:
`python main.py --dataset icews14_irr`

We kindly ask you not to distribute! Thank you!