import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import pickle as pkl
from utils import get_f1_by_bio
from utils import get_f1_by_bio_nomask
from utils import get_mention_f1
from getCluster import get_cluster
from utils import evaluate_coref
from scripts.utils import refind_entity
from metrics_back import CorefEvaluator

#from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
from pytorch_transformers import BertTokenizer

class myLSTM(nn.Module):
    """
    My lstm for text classification
    """
    def __init__(self, config, device):
        super(myLSTM, self).__init__()
        self.device = device
        self.hidden_size = config.hidden_size
        self.output_size = config.num_labels
        self.dropout = nn.Dropout(0.2)
        #self.pos_lstm = nn.LSTM()
        self.word_dict = config.word_dict

        hidden_size = self.hidden_size
        output_size = self.output_size + 1
        self.embedding = nn.Embedding(config.word_vocab, config.embedding_size)
        self.lstm = nn.LSTM(config.embedding_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.atten_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        hidden_size_low = 100
        
        #self.mention_linear = nn.Linear(hidden_size * 2, output_size + 1)
        self.mention_linear = nn.Sequential(
            nn.Linear( hidden_size_low+ config.pos_emb_size * 2 + config.sememe_emb_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        self.mention_linear_no_pos = nn.Sequential(
            nn.Linear(hidden_size_low, hidden_size_low),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_low, output_size)
        )
        
        
        self.linear = nn.Linear(hidden_size_low * 4, output_size - 1)
        self.linear_1 = nn.Linear(config.hidden_size * 2, hidden_size_low)

        self.gcn = GCN(hidden_size_low,hidden_size_low,self.dropout)

        self.fuse_1 = nn.Linear(hidden_size_low, hidden_size_low, bias=False)
        self.fuse_2 = nn.Linear(hidden_size_low, hidden_size_low)

    def get_label_f1(self, predict_indices, gold_indices):

        tp, fp, tn = 0, 0, 0

        predict_indices = set(predict_indices)
        gold_indices = set(gold_indices)

        for w in predict_indices:
            if w in gold_indices:
                tp += 1
        p = tp / len(predict_indices)
        r = tp / len(gold_indices)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        return f1

    def get_mention_indices(self, outputs):
        #lstm_out = self.mention_linear(lstm_out)
        if len(outputs.shape) > 2:
            outputs = outputs.argmax(-1)
            # outputs = [w.argmax(-1) for w in outputs]
        bio_dict = {'B': 0, 'I': 1, 'O':2}
        bio_dict = {v:k for k, v in bio_dict.items()}

        outputs = [[bio_dict[w.item()] for w in out] for out in outputs]
        #outputs = [bio_dict[w.item()] for w in outputs]

        indices = []
        length = 0
        for output in outputs:
            mention_index = []
            start, end = -1, -1
            for i, out in enumerate(output):
                index = i + length
                if out == 'B':
                    if start != -1:
                        mention_index.append((start, end))
                    start, end = index, index
                elif out == 'O':
                    if start != -1:
                        mention_index.append((start, end))
                    start, end = -1, -1
                else:
                    end = index
            if start != -1:
                mention_index.append((start, end))
            indices.append(mention_index)
        return indices
    
    def get_mention_emb(self, lstm_out, mention_index):
        mention_emb_list = []
        mention_start, mention_end = zip(*mention_index)
        mention_start = torch.tensor(mention_start).to(self.device)
        mention_end = torch.tensor(mention_end).to(self.device)
        mention_emb_list.append(lstm_out.index_select(0, mention_start))
        mention_emb_list.append(lstm_out.index_select(0, mention_end))
        mention_emb = torch.cat(mention_emb_list, 1)
        return mention_emb

    def get_mention_labels(self, predict_indices, gold_sets):

        mention_matrix = torch.zeros(len(predict_indices), len(predict_indices)).long().to(self.device)
        indices_dict = {w : i for i, w in enumerate(predict_indices)}
        for i in range(len(predict_indices)):
            mention_matrix[i, i] = 1
            pass
        for gold_set in gold_sets:
            for mention_0 in gold_set:
                if mention_0 not in indices_dict:
                    continue
                for mention_1 in gold_set:
                    if mention_1 not in indices_dict:
                        continue
                    s1, s2 = indices_dict[mention_0], indices_dict[mention_1]
                    mention_matrix[s1, s2] = 1
                    mention_matrix[s2, s1] = 1
        return mention_matrix
    
    def get_mask(self, lengths):
        tmp = lengths.cpu()
        return torch.arange(max(tmp))[None, :] < tmp[:, None]

    def get_mention_nomask(self, lstm_out, masks):
        if len(masks.shape) > 1:
            masks = masks.reshape([-1, ])
        masks = masks.nonzero().reshape(-1)

        return lstm_out[masks]

    def get_cluster(self, predict_indices, mention_label):
        if len(mention_label.shape) == 3:
            predict_indices = [tuple(w) for w in predict_indices]

        cluster = dict()

        for i in range(len(mention_label)):
            for j in range(i):
                if mention_label[i, j] == 1:
                    if predict_indices[j] not in cluster:
                        cluster[predict_indices[j]] = set()
                    if predict_indices[i] in cluster[predict_indices[j]]:
                        continue
                    cluster[predict_indices[j]].add(predict_indices[i])

    def refind_gold(self, input_ids, gold_indices, input_masks, mention_label):
        print("gold", mention_label)
        print("length", len(gold_indices))
        print("len mention", len(mention_label))
        if len(mention_label) > 13:
            return
        nomask = []
        sms = 0
        for i in range(len(input_masks)):
            sm = sum(input_masks[i])
            nomask += [j + sms for j in range(sm)][1:-1]
            sms += sm

        input_ids = input_ids.reshape(-1)
        input_ids = self.get_mention_nomask(input_ids, input_masks)
        input_ids = input_ids[nomask]

        result_html = pkl.load(open('result_html/res.pkl', 'rb'))
        #word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
        res = []
        res = input_ids.tolist()
        res = self.tokenizer.convert_ids_to_tokens(res)
        #clusters = mention_label
        clusters = [gold_indices]

        i, j = 0, 0
        result_text = '<p>'
        colors = ['red', 'green', 'blue', 'yellow']
        colors = ['red'] * 4

        result_text = '<p>'
        for index, cluster in enumerate(clusters):
            i,j = 0, 0
            predict_indices = cluster
            while i < len(res) and j < len(predict_indices):
                if i < predict_indices[j][0]:
                    result_text += res[i]
                elif i == predict_indices[j][0]:
                    result_text += '<font color="{}">'.format(colors[index % 4]) + res[i]
                    if predict_indices[j][1] == i:
                        j += 1
                        result_text += '</font>'
                elif i > predict_indices[j][0] and i < predict_indices[j][1]:
                    result_text += res[i]
                elif i == predict_indices[j][1]:
                    result_text += res[i] + '</font>'
                    j += 1
                i += 1
            while i < len(res):
                result_text += res[i]
                i += 1
            result_text += '</p>'
            result_text += '<p></p><p>'
        result_text += '</p>'
        result_html = result_html.format(result_text)
        open('result_html/res_gold.html', 'w').write(result_html)
        print("input...")
        a = input()
    def refind(self, input_ids, predict_indices, input_masks, gold='no'):
        nomask = []
        sms = 0
        for i in range(len(input_masks)):
            sm = sum(input_masks[i])
            nomask += [j + sms for j in range(sm)][1:-1]
            sms += sm

        input_ids = input_ids.reshape(-1)
        input_ids = self.get_mention_nomask(input_ids, input_masks)
        input_ids = input_ids[nomask]

        result_html = pkl.load(open('res.pkl', 'rb'))
        #word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
        res = []
        res = input_ids.tolist()
        res = self.tokenizer.convert_ids_to_tokens(res)

        #clusters = get_cluster(predict_indices, mention_label)
        clusters = predict_indices

        i, j = 0, 0
        result_text = '<p>'
        colors = ['fuchsia', 'yellow', 'aqua', 'lime']

        result_text = '<p>'
        for index, cluster in enumerate(clusters):
            i,j = 0, 0
            predict_indices = cluster
            while i < len(res) and j < len(predict_indices):
                if i < predict_indices[j][0]:
                    result_text += res[i]
                elif i == predict_indices[j][0]:
                    result_text += '<font style="background-color:{}">'.format(colors[index % 4]) + res[i]
                    if predict_indices[j][1] == i:
                        j += 1
                        result_text += '</font>'
                elif i > predict_indices[j][0] and i < predict_indices[j][1]:
                    result_text += res[i]
                elif i == predict_indices[j][1]:
                    result_text += res[i] + '</font>'
                    j += 1
                i += 1
            while i < len(res):
                result_text += res[i]
                i += 1
            result_text += '</p>'
            result_text += '<p></p><p>'
        result_text += '</p>'
        #result_text = result_text.replace('p>', 'h4>')
        if gold == 'g':
            a = str(open('res/{}.html'.format(self.valid_index), 'r').read())
            a = a.replace('ggg', '{}').format(result_text)
            open('res/{}.html'.format(self.valid_index), 'w').write(a)
        else:
            result_html = result_html.format(result_text, 'ggg')
            f = open('res/{}.html'.format(self.valid_index), 'w')
            f.write(result_html)
            f.flush()
            f.close()

    def check(self, input_ids, lengths, gold_mention):
        word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
        word_dict = {v:w for w, v in word_dict.items()}
        res = []
        for input_id, length in zip(input_ids, lengths):
            tmp = input_id[:length].cpu().tolist()
            print(tmp)
            tmp = self.tokenizer.convert_ids_to_tokens(tmp)
            res+= tmp
        print(res)
        gold_mention = [w for line in gold_mention for w in line]
        for ment in gold_mention:
            print(''.join(res[ment[0]:ment[1]+1]))
            print('input..')

    def get_f1_by_bio(self, predict_bio, gold_bio):
        pass

    def delete_head_tail(self, outputs, masks):
        lengths = [w.sum().item() for w in masks]
        out = []
        start = 0
        i = 0
        while start < outputs.shape[0]:
            if lengths[i] > 2:
                out.append(outputs[start+1:start + lengths[i]-1])
            start += lengths[i]
            i += 1
        return out

    
    def get_hownet_mask(self, lengths):
        # 20, 512, 30
        max_num = max([max(w) for w in lengths.cpu().tolist()])
        res = []
        for length in lengths:
            tmp = length.cpu()
            tmp = torch.arange(max_num)[None, :] < tmp[:, None]
            res.append(tmp.float())
            #res.append(tmp)
            #res.append(torch.stack(res0, 0))
        res = torch.stack(res, 0).to(self.device)
        return res

    def get_dense_mention(self, output : torch.Tensor, sentence_counts, input_lengths, input_labels, input_ids):
        input_res, label_res = [], []
        cumulative, current_sent_id = -1, 0
        cur_input, cur_label = [], []
        input_id_res = []
        cur_id = []

        for i in range(output.shape[0]):
            length = input_lengths[i]
            line = output[i][:length]
            cur_input.append(line)
            cur_label.append(input_labels[i][:length])
            cur_id.append(input_ids[i][:length])
            if i == sentence_counts[current_sent_id] + cumulative or i == output.shape[0] - 1:
                cumulative += sentence_counts[current_sent_id]
                current_sent_id += 1
                input_res.append(torch.cat(cur_input, 0))
                label_res.append(torch.cat(cur_label, 0))
                input_id_res.append(torch.cat(cur_id, 0))
                cur_input, cur_label = [], []
                cur_id = []
        document_lengths = sentence_counts.new_zeros(sentence_counts.shape[0])
        for i, w in enumerate(input_res):
            document_lengths[i] = w.shape[0]

        max_lengths = document_lengths.max().item()
        batch_input = output.new_zeros([sentence_counts.shape[0], max_lengths, output.shape[-1]])

        dense_mask = sentence_counts.new_zeros([document_lengths.shape[0], max_lengths])
        batch_label = output.new_zeros([sentence_counts.shape[0], max_lengths])
        batch_idx = sentence_counts.new_zeros([document_lengths.shape[0], max_lengths])
        for i, w in enumerate(input_res):
            batch_input[i][:w.shape[0]] = w
            batch_label[i][:w.shape[0]] = label_res[i]
            dense_mask[i][:w.shape[0]] = 1
            batch_idx[i][:w.shape[0]] = input_id_res[i]

        dense_mask = dense_mask.reshape(-1,)
        return batch_input, batch_label.long(), document_lengths, dense_mask, batch_idx


    def flatten_batch_output(self, input, input_lengths):
        input = [w for line in input for w in line]
        return input


    def forward(self, input_ids, input_lengths, input_labels, mention_sets, sentence_counts, reverse_sort, show_res=False, coref_evaluator=None):
        #mention_sets = [[(p[0][0].item(), p[1][0].item()) for p in k] for k in mention_sets if len(k) > 0]
        mention_sets = [[k for k in mention_set if len(k) > 0] for mention_set in mention_sets]

        input_embedding = self.embedding(input_ids)
        paded = nn.utils.rnn.pack_padded_sequence(input_embedding, input_lengths, batch_first=True)
        output, _ = self.lstm(paded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = output[reverse_sort]
        input_ids = input_ids[reverse_sort]
        input_lengths = input_lengths[reverse_sort]

        output = self.dropout(output)
        criterion = nn.CrossEntropyLoss(reduction='none')
        output = self.linear_1(output)
        dense_input, dense_labels, document_lengths, dense_mask, batch_idx = self.get_dense_mention(output, sentence_counts, input_lengths, input_labels, input_ids)

        predict_output = self.mention_linear_no_pos(dense_input)
        t = dense_labels.reshape(-1, )
        p = predict_output.reshape([-1, predict_output.shape[-1]])
        #losses = criterion(predict_output.reshape([-1, predict_output.shape[-1]]), dense_labels.reshape(-1, ))
        losses = (criterion(predict_output.reshape([-1, predict_output.shape[-1]]), dense_labels.reshape(-1, )) * dense_mask).mean()

        predict_indices = self.get_mention_indices(predict_output)
        gold_indices = self.get_mention_indices(dense_labels)
        for i in range(len(predict_indices)):
            if len(predict_indices[i]) == 0:
                predict_indices[i] = gold_indices[i][:1]
        #predict_indices = gold_indices
        criterion = nn.CrossEntropyLoss()
        total_count, correct_counts = [], []
        self.cluster = []
        for i in range(dense_input.shape[0]):
            mention_emb = self.get_mention_emb(dense_input[i], predict_indices[i])
            mention_label = self.get_mention_labels(predict_indices[i], mention_sets[i])

            mention_emb_r = mention_emb.unsqueeze(1)
            mention_emb_c = mention_emb.unsqueeze(0)

            mention_emb_agg = torch.cat((mention_emb_c * mention_emb_r, mention_emb_r + mention_emb_c), -1)

            mention_interaction = self.linear(mention_emb_agg)
            correct_count, count = 0, 0
            new_mention_interaction, new_mention_label = [], []
            for j in range(mention_interaction.shape[0]):
                new_mention_interaction.append(mention_interaction[j, :j + 1])
                new_mention_label.append(mention_label[j, :j + 1])
            new_mention_label = torch.cat(new_mention_label)
            new_mention_interaction = torch.cat(new_mention_interaction, 0)
            tmp_loss = criterion(new_mention_interaction, new_mention_label)
            #losses += tmp_loss

            self.predict_mention = new_mention_interaction.cpu().tolist()
            self.gold_mention = new_mention_label.cpu().tolist()


            pred_cluster, gold_cluster = evaluate_coref(predict_indices[i], mention_interaction, mention_sets[i], coref_evaluator)
            self.cluster.append((pred_cluster, gold_cluster))
            correct_counts.append(correct_count)
            total_count.append(count)

        #predict_output = self.flatten_batch_output(predict_output, document_lengths)
        #dense_labels = self.flatten_batch_output(dense_labels, document_lengths)
        #refind_entity(batch_idx, dense_labels, self.word_dict)

        return losses, correct_counts, total_count, predict_output, dense_labels, document_lengths
        tmp_prf = coref_evaluator.get_prf()

        self.cluster = (pred_cluster, gold_cluster)

        return losses, correct_count, count, predict_output,labels, new_mention_interaction.cpu().tolist(), new_mention_label.cpu().tolist()


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        #output = torch.spmm(adj, support)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 =GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        #x = self.dropout(x)
        x = torch.relu(self.gc1(x, adj))
        #x = self.dropout(x)
        #x = torch.relu(self.gc2(x, adj))
        x = self.dropout(x)
        #x = self.gc3(x, adj)
        #x = self.gc2(x, adj)
        return x
        return torch.log_softmax(x, dim=1)
