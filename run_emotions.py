import os
import random
import time
import datetime
import torch
import argparse
import numpy as np
import pandas as pd
from torch.nn import functional as F
from sklearn.metrics import (f1_score, recall_score, accuracy_score,
                                precision_score, jaccard_score,jaccard_similarity_score)
from transformers import (get_linear_schedule_with_warmup,AdamW,AutoModel, AutoTokenizer,
                            AutoModelForSequenceClassification)
from torch.utils.data import (TensorDataset,DataLoader,
                             RandomSampler, SequentialSampler, Dataset)




MODEL_CLASSES = {
    'bertweet' : 'vinai/bertweet-base',
    'roberta' : 'roberta-base',
    'bert' : 'bert-base-uncased'
}


def read_dataset(root_dir='.',dataset='hate'):
    if not os.path.exists(root_dir):
        return None
    with open(f'{root_dir}/tweeteval/datasets/{dataset}/train_text.txt') as train_tweets_file:
        train_tweets = train_tweets_file.readlines()
    with open(f'{root_dir}/tweeteval/datasets/{dataset}/train_labels.txt') as train_label_file:
        train_tweets_labels = train_label_file.readlines()
    with open(f'{root_dir}/tweeteval/datasets/{dataset}/val_text.txt') as valid_tweets_file:
        valid_tweets = valid_tweets_file.readlines()
    with open(f'{root_dir}/tweeteval/datasets/{dataset}/val_labels.txt') as valid_label_file:
        valid_tweets_labels = valid_label_file.readlines()
    with open(f'{root_dir}/tweeteval/datasets/{dataset}/test_text.txt') as test_tweets_file:
        test_tweets = test_tweets_file.readlines()
    with open(f'{root_dir}/tweeteval/datasets/{dataset}/test_labels.txt') as test_label_file:
        test_tweets_labels = test_label_file.readlines()

    df_train = pd.DataFrame({'tweet':train_tweets,'label':[int(item.strip('\n')) for item in train_tweets_labels]},columns=['tweet','label'])
    df_test = pd.DataFrame({'tweet':test_tweets,'label':[int(item.strip('\n')) for item in test_tweets_labels]},columns=['tweet','label'])
    df_valid = pd.DataFrame({'tweet':valid_tweets,'label':[int(item.strip('\n')) for item in valid_tweets_labels]},columns=['tweet','label'])
    num_classes = df_train.label.unique().shape[0]
    return df_train,df_test,df_valid,num_classes

def encode_sentence(s, tokenizer):
   return tokenizer.encode_plus(
                        s,                      
                        add_special_tokens = True, 
                        max_length = 128,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )

def bert_encode(df, tokenizer, max_seq_length=512):
    input_ids = []
    attention_masks = []
    for sent in df[["tweet"]].values:
        sent = sent.item()
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 128,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',    
                    )
           
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
    'input_word_ids': input_ids,
    'input_mask': attention_masks}

    return inputs

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def validate(model,test_dataloader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    preds = []
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    t0 = time.time()
    for batch in test_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        preds.append(logits)
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(test_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Test Loss: {0:.2f}".format(avg_val_loss))
    print("  Test took: {:}".format(validation_time))
    return preds, avg_val_accuracy, avg_val_loss, validation_time


def prepare_dataloaders(df_train,df_test,df_valid,tokenizer_class="vinai/bertweet-base",batch_size=32):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_class, use_fast=False, normalization=True)

    tweet_valid = bert_encode(df_valid, tokenizer)
    tweet_valid_labels = df_valid['label'].astype(int)
    tweet_train = bert_encode(df_train, tokenizer)
    tweet_train_labels = df_train['label'].astype(int)
    tweet_test = bert_encode(df_test, tokenizer)
    tweet_test_labels = df_test['label'].astype(int)


    input_ids, attention_masks = tweet_train.values()
    labels = torch.tensor(tweet_train_labels.values)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    input_ids, attention_masks = tweet_valid.values()
    labels = torch.tensor(tweet_valid_labels.values)
    val_dataset = TensorDataset(input_ids, attention_masks, labels)
    input_ids, attention_masks = tweet_test.values()
    labels = torch.tensor(tweet_test_labels.values)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    
    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )


    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size 
            )


    test_dataloader = DataLoader(
                test_dataset, 
                sampler = SequentialSampler(test_dataset), 
                batch_size = batch_size
            )
    return train_dataloader,validation_dataloader,test_dataloader

def prepare_model(model_class="vinai/bertweet-base",num_classes=2,model_to_load=None,total_steps=-1):



    model = AutoModelForSequenceClassification.from_pretrained(
        model_class,
        num_labels = num_classes,  
        output_attentions = False, 
        output_hidden_states = False,
    )

    optimizer = AdamW(model.parameters(),
                    lr = 5e-5,
                    eps = 1e-8
                    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

    if model_to_load is not None:
        try:
            model.roberta.load_state_dict(torch.load(model_to_load))
            print("LOADED MODEL")
        except:
            pass
    return model, optimizer, scheduler

def train(model,optimizer,scheduler,train_dataloader,validation_dataloader,epochs,save_location):
    seed_val = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        _, avg_val_accuracy, avg_val_loss, validation_time = validate(model,validation_dataloader)
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    torch.save(model.cpu().roberta.state_dict(),save_location)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32,help='The batch size for training')
    parser.add_argument('--epochs',type=int,default=4,help='The batch size for training')
    parser.add_argument('--total_steps',type=int,default=-1,help='Number of training steps')
    parser.add_argument('--dataset_location',type=str,default='.',help='The tweetEval dataset location')
    parser.add_argument('--model_class',type=str,default='bertweet', choices=['bertweet','bert','roberta'],help='The pre-trained hugginface model to load')
    parser.add_argument('--dataset',type=str,default='hate',choices=['emoji','emotion','hate','irony','offensive','sentiment'],help='The TweetEval dataset to choose')
    parser.add_argument('--model_to_load',type=str,default=None,help='Load pre-trained BERT')
    parser.add_argument('--save',type=str,default='./model.pb',help='Save the model to disk')

    args = parser.parse_args()

    df_train,df_test,df_valid,num_classes = read_dataset(root_dir=args.dataset_location,dataset=args.dataset)
    train_dataloader,validation_dataloader,test_dataloader = prepare_dataloaders(df_train,df_test,
                                                                df_valid,
                                                                tokenizer_class="vinai/bertweet-base",
                                                                batch_size=args.batch_size)
    if args.total_steps == -1 and train_dataloader is not None:
        args.total_steps = len(train_dataloader) * args.epochs
    elif args.total_steps == -1 and train_dataloader is None:
        args.total_steps = 1000

    model, optimizer, scheduler = prepare_model(MODEL_CLASSES[args.model_class],num_classes,args.model_to_load,args.total_steps)
    train(model,optimizer,scheduler,train_dataloader,validation_dataloader,args.epochs,args.save)
    validate(model,test_dataloader)


if __name__ == '__main__':
    main()

    




