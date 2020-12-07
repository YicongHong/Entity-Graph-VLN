import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import pdb

class ObjEncoder(nn.Module):
    ''' Encodes object labels using GloVe. '''

    def __init__(self, vocab_size, embedding_size, glove_matrix):
        super(ObjEncoder, self).__init__()

        padding_idx = 100
        word_embeds = nn.Embedding(vocab_size, embedding_size, padding_idx)
        word_embeds.load_state_dict({'weight': glove_matrix})
        self.embedding = word_embeds
        self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        return embeds


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        word_embeds = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.embedding = word_embeds

        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)

        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.encoder2graph = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)

    def init_state(self, inputs):
        # Initialize to zero cell states and hidden states
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths, obj_encode=False):
        # Expects input vocab indices as (batch, seq_len). Also requires a
        #    list of lengths for dynamic batching.

        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)

        if obj_encode:
            return embeds
        else:
            h0, c0 = self.init_state(inputs)
            packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
            if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
            graph_init = nn.Tanh()(self.encoder2graph(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
            graph_init = nn.Tanh()(self.encoder2graph(h_t))
        else:
            assert False

        ctx = self.drop(ctx)

        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t), zeros_like(graph_init)
        else:
            return ctx, decoder_init, c_t, graph_init


class AttnDecoderLSTM_Graph(nn.Module):
    ''' The Vision-and-Language Entity Relationship Graph. '''

    def __init__(self, embedding_size, hidden_size, dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM_Graph, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(nn.Linear(args.angle_feat_size, self.embedding_size), nn.Tanh())

        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)

        self.lstm_G = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.top_N_obj = args.n_objects
        self.object_raw_dim = 300
        self.action_proj = nn.Sequential(nn.Linear(args.angle_feat_size, hidden_size, bias=False), nn.Tanh())

        self.attention_layer_S = SoftDotAttention(hidden_size, hidden_size)
        self.attention_layer_O = SoftDotAttention(hidden_size, hidden_size)
        self.attention_layer_A = SoftDotAttention(hidden_size, hidden_size)

        self.message_attn_so = SoftDotAttention(hidden_size, hidden_size)
        self.message_attn_as = SoftDotAttention(hidden_size, hidden_size)
        self.message_attn_oa = SoftDotAttention(hidden_size, hidden_size)

        self.cand_obj_attn = SoftObjAttention(hidden_size, self.object_raw_dim)

        self.h_a_s_fc = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())
        self.h_s_o_fc = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())
        self.h_o_a_fc = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())

        self.sp = nn.Sequential(nn.Linear(feature_size-args.angle_feat_size, hidden_size, bias=False), nn.Tanh())
        self.op = nn.Sequential(nn.Linear(self.object_raw_dim, hidden_size, bias=False), nn.Tanh())
        self.ap = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())

        self.si = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())
        self.oi = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())
        self.ai = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())

        self.a_s = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=True), nn.Tanh())
        self.s_o = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=True), nn.Tanh())
        self.o_a = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=True), nn.Tanh())

        self.ms = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())
        self.mo = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())
        self.ma = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, bias=False), nn.Tanh())

        self.logit_fc = nn.Linear(hidden_size*3, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, action, feature, cand_feat, cand_object_feat,
        h_0, prev_h1, c_0, h_graph, ctx, ctx_mask=None, obs=None, train_rl=None, already_dropfeat=False):

        ''' previous action direction encoding '''
        action_embeds = self.embedding(action); action_embeds = self.drop(action_embeds)

        cand_obj_ori_feat = self.drop_env(cand_object_feat)

        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        ''' global visual attention '''
        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_global_feat = torch.cat((action_embeds, attn_feat), 1)
        h_1, c_1 = self.lstm_G(concat_global_feat, (prev_h1, c_0))
        h_1_drop = self.drop(h_1)

        ''' laguage attention '''
        h_tilde_S, alpha_S = self.attention_layer_S(h_1_drop, ctx, ctx_mask, output_tilde=False)
        h_tilde_O, alpha_O = self.attention_layer_O(h_1_drop, ctx, ctx_mask, output_tilde=False)
        h_tilde_A, alpha_A = self.attention_layer_A(h_1_drop, ctx, ctx_mask, output_tilde=False)
        h_tilde_Expert = (h_tilde_S + h_tilde_O + h_tilde_A) / 3.0
        h_tilde_drop_S = self.drop(h_tilde_S); h_tilde_drop_O = self.drop(h_tilde_O); h_tilde_drop_A = self.drop(h_tilde_A)

        ''' candidate global feature '''
        cand_vis_feat = cand_feat[..., :-args.angle_feat_size]
        ''' candidate object feature attention at single view '''
        cand_obj_feat, _ = self.cand_obj_attn(h_tilde_O, cand_obj_ori_feat)
        ''' candidate action feature '''
        action_feat = self.action_proj(cand_feat[..., -args.angle_feat_size:])

        ''' structural - visual feature context '''
        context_s = torch.repeat_interleave(h_tilde_drop_S.unsqueeze(1), cand_feat.size(1), dim=1)
        context_o = torch.repeat_interleave(h_tilde_drop_O.unsqueeze(1), cand_feat.size(1), dim=1)
        context_a = torch.repeat_interleave(h_tilde_drop_A.unsqueeze(1), cand_feat.size(1), dim=1)

        ''' structural - message context '''
        h_s_o = self.h_s_o_fc(torch.cat((h_tilde_drop_S, h_tilde_drop_O), dim=1))
        h_a_s = self.h_a_s_fc(torch.cat((h_tilde_drop_A, h_tilde_drop_S), dim=1))
        h_o_a = self.h_o_a_fc(torch.cat((h_tilde_drop_O, h_tilde_drop_A), dim=1))

        ctx_s_o, beta_so = self.message_attn_so(h_s_o, ctx, ctx_mask, output_tilde=False)
        ctx_a_s, beta_as = self.message_attn_as(h_a_s, ctx, ctx_mask, output_tilde=False)
        ctx_o_a, beta_oa = self.message_attn_oa(h_o_a, ctx, ctx_mask, output_tilde=False)
        ctx_s_o = self.drop(ctx_s_o); ctx_a_s = self.drop(ctx_a_s); ctx_o_a = self.drop(ctx_o_a)

        ctx_mso = torch.repeat_interleave(ctx_s_o.unsqueeze(1), cand_feat.size(1), dim=1)
        ctx_mas = torch.repeat_interleave(ctx_a_s.unsqueeze(1), cand_feat.size(1), dim=1)
        ctx_moa = torch.repeat_interleave(ctx_o_a.unsqueeze(1), cand_feat.size(1), dim=1)

        ''' node initialization '''
        h_graph = self.drop_env(h_graph)
        h_graph_rep = torch.repeat_interleave(h_graph.unsqueeze(1), cand_feat.size(1), dim=1) # temporal link
        scene_tilde_p = self.sp(cand_vis_feat)
        object_tilde_p = self.op(cand_obj_feat)
        action_tilde_p = self.ap(torch.cat((h_graph_rep, action_feat), dim=2))

        scene_tilde_i = self.si(context_s * scene_tilde_p)
        object_tilde_i = self.oi(context_o * object_tilde_p)
        action_tilde_i = self.ai(context_a * action_tilde_p)
        scene_tilde_0 = scene_tilde_i.clone(); object_tilde_0 = object_tilde_i.clone(); action_tilde_0 = action_tilde_i.clone()

        ''' structural - message passing '''
        m_os = self.s_o(torch.cat((object_tilde_0, scene_tilde_0), dim=2))
        m_as = self.a_s(torch.cat((action_tilde_0, scene_tilde_0), dim=2))
        layernorm_m = nn.LayerNorm(m_as.size()[1:], elementwise_affine=False)  # could change to something like BertLayerNorm
        m_os = layernorm_m(m_os); m_as = layernorm_m(m_as)
        scene_tilde_1 = self.ms(torch.cat((m_os*ctx_mso, m_as*ctx_mas), dim=2)) + scene_tilde_i
        scene_tilde_1 = self.drop_env(scene_tilde_1)

        m_so = self.s_o(torch.cat((scene_tilde_0, object_tilde_0), dim=2))
        m_ao = self.o_a(torch.cat((action_tilde_0, object_tilde_0), dim=2))
        m_so = layernorm_m(m_so); m_ao = layernorm_m(m_ao)
        object_tilde_1 = self.mo(torch.cat((m_so*ctx_mso, m_ao*ctx_moa), dim=2)) + object_tilde_i
        object_tilde_1 = self.drop_env(object_tilde_1)

        m_oa = self.o_a(torch.cat((object_tilde_0, action_tilde_0), dim=2))
        m_sa = self.a_s(torch.cat((scene_tilde_0, action_tilde_0), dim=2))
        m_oa = layernorm_m(m_oa); m_sa = layernorm_m(m_sa)
        action_tilde_1 = self.ma(torch.cat((m_oa*ctx_moa, m_sa*ctx_mas), dim=2)) + action_tilde_i
        action_tilde_1 = self.drop_env(action_tilde_1)

        logit = self.logit_fc(torch.cat((scene_tilde_1, object_tilde_1, action_tilde_1), dim=2)).squeeze()

        return h_1, c_1, logit, h_tilde_Expert, action_tilde_1


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim , args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).

        Code: https://github.com/huggingface/transformers
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, weighted_context, attn
        else:
            return weighted_context, attn


class SoftObjAttention(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftObjAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()

    def forward(self, h, context, mask=None):

        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        batch_size = context.size(0); max_cand = context.size(1)

        target = torch.repeat_interleave(target, context.size(1), dim=0)
        context = context.view(batch_size*max_cand, context.size(2), context.size(3))

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        attn = logit

        weighted_context = weighted_context.view(batch_size, max_cand, context.size(-1))
        attn = attn.view(batch_size, max_cand, attn.size(-1))

        return weighted_context, attn


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1
