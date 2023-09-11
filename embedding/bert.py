import datetime

import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
# from pytorch_transformers import BertModel

class CXTEBD(nn.Module):
    """
        An embedding layer directly returns precomputed BERT
        embeddings.
    """
    def __init__(self, args):
        """
            pretrained_model_name_or_path, cache_dir: check huggingface's codebase for details
            finetune_ebd: finetuning bert representation or not during
            meta-training
            return_seq: return a sequence of bert representations, or [cls]
        """
        super(CXTEBD, self).__init__()

        self.args = args

        print("{}, Loading pretrainedModel bert".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

        self.model = BertModel.from_pretrained("bert-base-uncased")
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size

    def get_bert(self, data):
        '''
            Return the last layer of bert's representation
            @param: bert_id: batch_size * max_text_len+2
            @param: text_len: text_len

            @return: last_layer: batch_size * max_text_len
        '''
        bert_id = data['text']
        text_len = data['text_len']

        len_range = torch.arange(bert_id.size()[-1], device=bert_id.device,
                                 dtype=text_len.dtype).expand(*bert_id.size())

        # mask for the bert
        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        # mask for the sep
        mask2 = (len_range < (text_len-1).unsqueeze(-1)).float().unsqueeze(-1)

        # need to use smaller batches
        out = self.model(bert_id, attention_mask=mask1)

        last_layer = mask2 * out[0]

        return last_layer


    def forward(self, data):

        with torch.no_grad():
            return self.get_bert(data)

