# BERT_for_twitter_performance_investigation
## Description
I investigated the performance of BERT/BERTweet performance on the TweetEval benchmark using different transfer learning scenarios. 

## Dataset:  
The dataset used in the TweetEval dataset at [tweetevel](https://github.com/cardiffnlp/tweeteval). To download the dataset: 
```bash
git clone https://github.com/cardiffnlp/tweeteval
```

## Dependancies
To install dependancies run the following command:
```bash
cd BERT_for_twitter_performance_investigation &&  pip install -r requirements.txt
```

## Training:
Here is the syntax of the python file
```
usage: run_emotions.py [-h] [--batch_size BATCH_SIZE]
                                               [--epochs EPOCHS]
                                               [--total_steps TOTAL_STEPS]
                                               [--dataset_location DATASET_LOCATION]
                                               [--model_class MODEL_CLASS]
                                               [--dataset {emoji,emotion,hate,irony,offensive,sentiment}]
                                               [--model_to_load MODEL_TO_LOAD]
                                               [--save SAVE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        The batch size for training
  --epochs EPOCHS       The batch size for training
  --total_steps TOTAL_STEPS
                        Number of training steps
  --dataset_location DATASET_LOCATION
                        The tweetEval dataset location
  --model_class MODEL_CLASS
                        The pre-trained hugginface model to load
  --dataset {emoji,emotion,hate,irony,offensive,sentiment}
                        The TweetEval dataset to choose
  --model_to_load MODEL_TO_LOAD
                        Load pre-trained BERT
  --save SAVE           Save the model to disk
```

Model is by default  [vinai/bertweet-base](https://huggingface.co/vinai/bertweet-base)

## Results


| Model | Sentiment [1] | Emotion [2] | Hate [3] | Irony [4] | Offensive [5] | Emoji [6] | Total |
|----------|------:|--------:|-----:|------:|----------:|----------:|---------|
| BERTweet   | 70.79       | **85.18**       | **59.06**    |82.62     | 83.25         | 38.90     | **69.92**     |
| BERTweet -> [2]  | 70.91      | 84.35       | 56.76   | 79.50    | 83.37         | **39.21**        | 69.01     |
| BERTweet -> [2] -> [3] | **70.95**     | 84.59       | 56.12    | 83.12     | **83.71**        | 39.12         | 69.60     |
| BERTweet -> [2] -> [3] -> [4] |70.33     | 83.79       | 55.44    | **83.62**     | **83.71**         | 38.91         | 69.3     |

### Note:
`Model -> [x]` means that model  was fine-tuned on dataset x.

## Reference:
This code was based on [This code](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

## TODO:
[ ] test all the compatible hugginface models
