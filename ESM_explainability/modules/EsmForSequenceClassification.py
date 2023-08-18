from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm import EsmPreTrainedModel

from ESM_explainability.modules.utils import *
from ESM_explainability.modules.ESM import EsmModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
from typing import List
import torch

from ESM_explainability.modules.utils import PaddedSequence


class FinetuneHeadModel(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = EsmClassificationHead(config)

        self.init_weights()
        self.tmp = {}

    def forward(
            self,
            esm_embedding,
    ):
        return self.classifier(self.dropout(esm_embedding))


class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.out_proj = Linear(config.hidden_size, config.num_labels)
        self.activation = Tanh()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS]) What ???
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.out_proj.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.activation.relprop(cam, **kwargs)
        cam = self.dense.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        # TODO: selection relprop?
        return cam


class ESMForSequenceClassification(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.esm = EsmModel(config)
        self.classifier = EsmClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = outputs[0]
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.esm.relprop(cam, **kwargs)
        return cam


class EsmClassifier(nn.Module):
    """Thin wrapper around EsmForSequenceClassification"""

    def __init__(self,
                 esm_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 num_labels: int,
                 max_length: int = 512,
                 use_half_precision=True):
        super(EsmClassifier, self).__init__()
        esm = ESMForSequenceClassification.from_pretrained(esm_dir, num_labels=num_labels)
        if use_half_precision:
            esm = esm.half()
        self.esm = esm
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(
            self,
            query: List[torch.tensor],
            document_batch: List[torch.tensor]
    ):
        assert len(query) == len(document_batch)
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        esm_input = PaddedSequence.autopad(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
            device=target_device
        )
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        (classes,) = self.esm(
            esm_input.data,
            attention_mask=esm_input.mask(on=0.0, off=float('-inf'), device=target_device),
            position_ids=positions.data
        )
        assert torch.all(classes == classes)  # for nans

        return classes

    def relprop(self, cam=None, **kwargs):
        return self.esm.relprop(cam, **kwargs)
