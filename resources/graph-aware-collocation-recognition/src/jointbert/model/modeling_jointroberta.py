import torch
import torch.nn as nn
from .module import IntentClassifier, SlotClassifier
from transformers import RobertaModel, RobertaConfig
from model.modeling_g2groberta import initialize_robertagraph


class JointRoberta(nn.Module):
    def __init__(self, args, config=None, intent_label_lst=None, slot_label_lst=None):
        super(JointRoberta, self).__init__()
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.use_g2g = self.args.use_g2g
        self.config = config
        if self.args.use_g2g:
            self.roberta = initialize_robertagraph(args.model_name_or_path, {'use_two_attn': self.args.use_two_attn,
                                                                             'just_attn': self.args.just_attn,
                                                                             "label_size": self.args.size_dep_label,
                                                                             "pos_size": self.args.pos_size})
        else:
            self.roberta = RobertaModel.from_pretrained(args.model_name_or_path)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            # self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
            raise NotImplementedError("CRF not implemented yet")

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids,
                pos_ids=None, graph_dep=None):  # sequence_output, pooled_output, (hidden_states), (attentions)

        if self.use_g2g:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, pos_ids=pos_ids,
                                   graph_arc=graph_dep)
        else:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here
        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
