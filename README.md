This is the code for the paper **A Simple But Powerful Graph Encoder for Temporal Knowledge Graph Completion** accepted to NeurIPS 2022 Temporal Graph Learning Workshop [paper](https://openreview.net/forum?id=DYG8RbgAIo)

### Preprocessing
If you want you run the demo on ICEWS14, ICEWS05-15 or GDELT, go to `'Software/dataset/${DATASET}'`, and run:
`python preprocess.py`
Then you can train on corresponding datasets.

### Training
To run the training demo, please run:
`python main.py --dataset ${DATASET}`

### Testing
To test a trained model, please run:
`python main.py --dataset ${DATASET} --test --resume --name ${CHECKPOINT_NAME}`

### Generalization to unseen timestamps
Please go to `'Software/dataset/icews14_unseen'` and run:
`python preprocess_extrapolate.py`
Then go to the root directory and run:
`python main.py --dataset icews14_unseen`

### Generalization to irregular timestamped data
Please run:
`python main.py --dataset icews14_irr`
