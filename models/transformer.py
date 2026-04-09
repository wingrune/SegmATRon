import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.gpt import GPT

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP2(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)


class SemanticTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_feature_embedding = nn.Linear(config.MODEL.IMG_FEATURE_SIZE, config.MODEL.EMBEDDING_DIM)
        self.prediction_embedding = MLP(config.MODEL.MASK_EMB_SIZE + config.MODEL.NUM_CLASSES + 1, 512, config.MODEL.EMBEDDING_DIM, 3)
        self.mask_embed = MLP(config.MODEL.OUTPUT_SIZE, 512, 256, 3)
        self.logit_decoder = nn.Linear(config.MODEL.OUTPUT_SIZE, config.MODEL.NUM_CLASSES + 1)
        self.loss_decoder = MLP(config.MODEL.OUTPUT_SIZE, 512, 1, 3)
        self.action_decoder = MLP(config.MODEL.OUTPUT_SIZE, 512, 5, 3)
        self.action_tokens = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, 5, config.MODEL.EMBEDDING_DIM),
                                                                   a=math.sqrt(5)))
        self.model = GPT(config.MODEL)
        self.num_actions = config.MODEL.NUM_ACTIONS


    def forward(self, x):

        img_feature_embedding = self.img_feature_embedding(x["embedded_memory_features"].permute(0, 1, 3, 4, 2))
        b, s, n, h, w = x["pred_masks"].shape
        preds = torch.cat((x["mask_features"], x["pred_logits"]), dim=-1)
        prediction_embeddings = self.prediction_embedding(preds) # B X S X P X N
        b, s, p, n = prediction_embeddings.shape
        n_preds = prediction_embeddings.shape[1] * prediction_embeddings.shape[2]
        seq = torch.cat((img_feature_embedding.reshape(b, -1, n),
                         prediction_embeddings.reshape(b, -1, n),
                         self.action_tokens.repeat(b, 1, 1).reshape(b, -1, n)), dim=1)
        y = self.model(seq)
        # unfold data
        y_preds = y[:, -(n_preds + self.num_actions):-self.num_actions].reshape(b, s, p, -1)
        mask_embed = self.mask_embed(y_preds)
        mask_features_key = "maskdino_mask_features" if "maskdino_mask_features" in x.keys() else "mask2former_mask_features"
        masks = torch.einsum("bspc,bschw->bsphw", mask_embed, x[mask_features_key])
        logits = self.logit_decoder(y_preds)
        loss = self.loss_decoder(y_preds)
        if self.num_actions > 1:
            actions = self.action_decoder(y[:, -self.num_actions:-1].reshape(b, self.num_actions-1,-1))
        else:
            actions = self.action_decoder(y[:, -2:-1].reshape(b, 1,-1))
        return {"seq": y_preds.squeeze(), "pred_masks": masks.squeeze(), "pred_logits": logits.squeeze(),
                "loss": loss, "actions": actions.squeeze()}

    def get_optimizer_groups(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('model.pos_emb')
        no_decay.add('action_tokens')
        no_decay.add('model.seq_pos_embed')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.WEIGHT_DECAY},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups