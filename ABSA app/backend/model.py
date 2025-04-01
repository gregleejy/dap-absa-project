from transformers import BertModel
from transformers import get_scheduler

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import os
from tqdm import tqdm

from transformers import BertModel
from transformers import get_scheduler

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

class ABSADataset_MTL(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length  # Limit sequence length to 512

    def __getitem__(self, idx):
        # return the values of token, tags, pol which for the row at idx as str data
        tokens, tags, pols = self.df.loc[idx, ['Words', 'Tags', 'Polarities']].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_tags = []
        bert_pols = []
        pols_label = 0

        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i]) # we tokenize each word in the token list
            bert_tokens += t
            # these two steps ensure that the bert_tags is the same length as the bert_tokens list
            bert_tags += [int(tags[i])] * len(t)
            bert_pols += [int(pols[i])] * len(t) #change labels from [-1-2] -> [0-3]

        # Truncate if exceeding max_length
        if len(bert_tokens) > self.max_length - 2:  # Account for [CLS] & [SEP]
            bert_tokens = bert_tokens[:self.max_length - 2]
            bert_tags = bert_tags[:self.max_length - 2]
            bert_pols = bert_pols[:self.max_length - 2]

        # Add special tokens [CLS] and [SEP]
        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        bert_tags = [0] + bert_tags + [0]  # Padding with neutral tag
        bert_pols = [0] + bert_pols + [0]  # Padding with neutral polarity

        # Convert tokens to IDs
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        # Convert lists to tensors
        ids_tensor = torch.tensor(bert_ids, dtype=torch.long)
        tags_tensor = torch.tensor(bert_tags, dtype=torch.long)
        pols_tensor = torch.tensor(bert_pols, dtype=torch.long)

        return bert_tokens, ids_tensor, tags_tensor, pols_tensor



    def __len__(self):
        return len(self.df)


class ABSABert_MTL(torch.nn.Module):
    def __init__(self, pretrain_model, adapter=True):
        super(ABSABert_MTL, self).__init__()
        self.adapter = adapter
        #removed the adapter functionality of the model as cannot import bertadaptermodel
        # if adapter:
        #     from transformers.adapters import BertAdapterModel
        #     self.bert = BertAdapterModel.from_pretrained(pretrain_model)
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.abte_head = torch.nn.Linear(self.bert.config.hidden_size, 2) #2 classes for 0,1 (None, Aspect)
        self.absa_head = torch.nn.Linear(self.bert.config.hidden_size, 4) #3 classes for 0,1,2,3 (None, Neg, Neu, Pos)


    def forward(self, ids_tensors, masks_tensors, segments_tensors = None):
        out_dict = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors)
        bert_outputs = out_dict['last_hidden_state']
        abte_outputs = self.abte_head(bert_outputs)
        absa_outputs = self.absa_head(bert_outputs)
        return abte_outputs, absa_outputs


class ABSAModel_MTL ():
    def __init__(self, tokenizer, adapter=True):
        self.model = ABSABert_MTL('bert-base-uncased', adapter=adapter)
        self.tokenizer = tokenizer
        self.trained = False
        self.adapter = adapter

    def padding(self, samples): #nothing to change here as we are just ensuring the sequence lengths are the same across sample in batches

        #bert_tokens, ids_tensor_abte, ids_tensor_absa, tags_tensor, pols_tensor, segment_tensor

        from torch.nn.utils.rnn import pad_sequence
        # Here s[0] for s in samples is the bert_tokens as defined in the __getitem__ method defined above
        ids_tensors_abte = [s[1] for s in samples]
        #pad_sequence ensures that all id_tensors in the list are of the same length
        ids_tensors_abte = pad_sequence(ids_tensors_abte, batch_first=True)
        # batch_first = True retunrs tensor in the shape of B x T x * where T is the longest sequence in the batch and B is the batch size
        # otherwise return in the shape of T x B x *

        tags_tensors = [s[2] for s in samples]
        tags_tensors = pad_sequence(tags_tensors, batch_first=True)

        pols_tensors = [s[3] for s in samples]
        pols_tensors = pad_sequence(pols_tensors, batch_first=True)

        masks_tensors_abte = torch.zeros(ids_tensors_abte.shape, dtype=torch.long)
        masks_tensors_abte = masks_tensors_abte.masked_fill(ids_tensors_abte != 0, 1) #this sets all real tokens to 1 and all padding to zero

        return ids_tensors_abte, tags_tensors, pols_tensors, masks_tensors_abte

    def loss_func(self, abte_logits, absa_logits, labels_abte, labels_absa):
        criterion = torch.nn.CrossEntropyLoss()

        #this flattens the tensor into a 1D vector for cross entropy loss computation
        labels_abte = labels_abte.view(-1)
        abte_logits = abte_logits.view(-1,2)
        labels_absa = labels_absa.view(-1)
        absa_logits = absa_logits.view(-1,4)

        lossABTE = criterion(abte_logits, labels_abte)
        lossABSA = criterion(absa_logits, labels_absa)

        return lossABTE+lossABSA

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path), strict=False)

    def save_model(self, model, name):
        torch.save(model.state_dict(), name)

    def train(self, data, epochs, device, path, batch_size=32, lr=1e-5, load_model=None, lr_schedule=True):

        #load model if lead_model is not None
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
                self.trained = True
            else:
                print("lead_model not found")

        # dataset and loader
        ds = ABSADataset_MTL(data, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self.padding)

        self.model = self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr) #optimzier takes in model parameters and learning rate as arg
        num_training_steps = epochs * len(loader)
        if lr_schedule: lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.losses = []

        all_data = len(loader)-1
        for epoch in range(epochs):
            finish_data = 0
            current_times = []
            n_batches = int(len(data)/batch_size)

            if self.adapter:
                if lr_schedule: dir_name  = path + "_" + "model_ABTE_MTL_adapter_scheduler"
                else: dir_name = path + "_" + "model_ABTE_MTL_adapter"
            else:
                if lr_schedule: dir_name  = path + "_" + "model_ABTE_MTL_scheduler"
                else: dir_name = path + "_" + "model_ABTE_MTL"

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            for nb in range((n_batches)):
                t0 = time.time()

                # ids_tensors, tags_tensors, _, masks_tensors = next(iter(loader))
                # ids_tensor = ids_tensors.to(device)
                # tags_tensor = tags_tensors.to(device)
                # masks_tensor = masks_tensors.to(device)

                # Fetch a batch
                ids_tensors_abte, tags_tensors, pols_tensors, masks_tensors_abte = next(iter(loader))

                # Move tensors to the appropriate device
                ids_tensors_abte = ids_tensors_abte.to(device)
                tags_tensors = tags_tensors.to(device)
                pols_tensors = pols_tensors.to(device)
                masks_tensors_abte = masks_tensors_abte.to(device)

                abte_outputs, absa_outputs = self.model(ids_tensors=ids_tensors_abte, masks_tensors=masks_tensors_abte)

                loss = self.loss_func(abte_outputs, absa_outputs, tags_tensors, pols_tensors)
                self.losses.append(loss.item()) #loss item returns the scalar value of the loss object
                loss.backward()
                optimizer.step()
                if lr_schedule: lr_scheduler.step()
                optimizer.zero_grad()

                finish_data += 1
                current_time = round(time.time() - t0,3)
                current_times.append(current_time)
                print("epoch: {}\tbatch: {}/{}\tloss: {}\tbatch time: {}\ttotal time: {}"\
                    .format(epoch, finish_data, all_data, loss.item(), current_time, sum(current_times)))

                np.savetxt('{}/losses_lr{}_epochs{}_batch{}.txt'.format(dir_name, lr, epochs, batch_size), self.losses)

            self.save_model(self.model, '{}/model_lr{}_epochs{}_batch{}.pkl'.format(dir_name, lr, epoch, batch_size))
            self.trained = True

    def history (self):
        if self.trained:
            return self.losses
        else:
            raise Exception('Model not trained')

    def predict(self, sentence, load_model=None, device='cpu'):
         # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

        # word_pieces = list(self.tokenizer.tokenize(sentence))
        # ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        # input_tensor = torch.tensor([ids]).to(device)

        MAX_LEN = 512
        word_pieces = self.tokenizer.tokenize(sentence)
        if len(word_pieces) > MAX_LEN-2:
            word_pieces = word_pieces[:MAX_LEN-2]

        word_pieces = ["[CLS]"] + word_pieces + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(device)


        #predict
        with torch.no_grad():
            abte_outputs, absa_outputs = self.model(input_tensor, None) #plug token ids into bert model
            _, predictions_abte = torch.max(abte_outputs, dim=2)
            _, predictions_absa = torch.max(absa_outputs, dim=2)
            # dim parameter = 0 operates at batch level, dim = 1 operates at sequence level, dim = 2 operates at token level
            # Since we are identifying aspects and their polarities here we will look at dim = 2

        predictions_abte = predictions_abte[0].tolist()
        predictions_absa = predictions_absa[0].tolist()

        return word_pieces, predictions_abte, predictions_absa, abte_outputs, absa_outputs

    def predict_batch(self, data, load_model=None, device='cpu'):

        tags_real = [t.strip('][').split(', ') for t in data['Tags']]
        tags_real = [[int(i) for i in t ] for t in tags_real]

        polarity_real = [t.strip('][').split(', ') for t in data['Polarities']]
        polarity_real = [[int(i) for i in p ] for p in polarity_real]
        # if -1 is not an aspect term, if 0 negative, if 2 positive, if 1 neutral, shift of 1
        """ Class labels start at 0 (important for ML models like CrossEntropyLoss on pytorch).
            -1 (not an aspect term) is converted to None to be ignored. The model receives
            properly formatted class labels. """
        # polarity_real = [[int(i)+1 for i in t] for t in polarity_real] old code

        # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

        predictions_abte = []
        predictions_absa = []

        for i in tqdm(range(len(data))):
            sentence = data['Words'][i]
            sentence = sentence.replace("'", "").strip("][").split(', ')
            sentence = ' '.join(sentence) #this converts the list back into a clean sentence
            w, p_abte, p_absa, _, _ = self.predict(sentence, load_model=load_model, device=device)
            predictions_abte.append(p_abte)
            predictions_absa.append(p_absa)
            tags_real[i] = tags_real[i][:len(p_abte)]
            polarity_real[i] = polarity_real[i][:len(p_absa)]

        return predictions_abte, predictions_absa, tags_real, polarity_real

    def _accuracy (self, x,y):
        return np.mean(np.array(x) == np.array(y))

    def test(self, dataset, load_model=None, device='cpu'): # device = 'cpu'
        from sklearn.metrics import classification_report
        # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

         # dataset and loader
        ds = ABSADataset_MTL(dataset, self.tokenizer)
        loader = DataLoader(ds, batch_size=50, shuffle=True, collate_fn=self.padding)

        pred_abte = []#padded list
        trueth_abte = [] #padded list
        pred_absa = []#padded list
        trueth_absa = [] #padded list
        with torch.no_grad(): # disables gradient calculations to save computaional memory
            for data in tqdm(loader): #tqdm takes in a iterable to show the progress

                # ids_tensors, tags_tensors, _, masks_tensors = data
                # ids_tensors = ids_tensors.to(device)
                # tags_tensors = tags_tensors.to(device)
                # masks_tensors = masks_tensors.to(device)
                ids_tensors, tags_tensors_abte, pols_tensors_absa, masks_tensors = data
                ids_tensors = ids_tensors.to(device)
                tags_tensors_abte = tags_tensors_abte.to(device)
                pols_tensors_absa = pols_tensors_absa.to(device)
                masks_tensors = masks_tensors.to(device)

                # outputs = self.model(ids_tensors=ids_tensors, masks_tensors=masks_tensors)
                abte_outputs, absa_outputs = self.model(ids_tensors=ids_tensors, masks_tensors=masks_tensors)

                _, p_abte = torch.max(abte_outputs, dim=2) #return the aspect class with the highest probability
                _, p_absa = torch.max(absa_outputs, dim=2) #return the polarity class with the highest probability

                # pred += list([int(j) for i in p for j in i ])
                # trueth += list([int(j) for i in tags_tensors for j in i ])

                #Flattens predictions (p_abte) and true labels (tags_tensors_abte) into lists, do the same for p_absa and pols_tensor_absa:
                pred_abte += [int(j) for i in p_abte for j in i]
                trueth_abte += [int(j) for i in tags_tensors_abte for j in i]

                pred_absa += [int(j) for i in p_absa for j in i]
                trueth_absa += [int(j) for i in pols_tensors_absa for j in i]

        # acc = self._accuracy(pred, trueth)
        # class_report = classification_report(trueth, pred, target_names=['none', 'start of AT', 'mark of AT'])

        acc_abte = self._accuracy(pred_abte, trueth_abte)
        class_report_abte = classification_report(trueth_abte, pred_abte, target_names=['none', 'AT'])

        acc_absa = self._accuracy(pred_absa, trueth_absa)
        class_report_absa = classification_report(trueth_absa, pred_absa, target_names=['none', 'negative', 'neutral', 'positive'])

        return acc_abte, class_report_abte, acc_absa, class_report_absa

    def accuracy(self, data, load_model=None, device='cpu'): #kinda useless function as it js calls self.test
      # def accuracy(self, data, load_model=None, device='cpu'):
        abte_acc, _, absa_acc, _  = self.test(data, load_model=load_model, device=device)
        return abte_acc, absa_acc