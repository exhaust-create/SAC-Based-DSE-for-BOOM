import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# Critic
class Q_Critic_Attention(nn.Module):
    """
    Returns
    -------
    q_value_of_options:
        The Q values of all options. The shape is (batch_size, num_options).
    """
    def __init__(self, embed_dim, num_attention_layers, num_heads, fnn_dim=None, device='cpu'):
        super(Q_Critic_Attention, self).__init__()
        fnn_dim = 2*embed_dim if fnn_dim is None else fnn_dim
        # Generate attention layers according to `num_attention_layers`
        self.attention_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_attention_layers)])
        self.attention_layers = nn.ModuleList([MultiheadAttention(embed_dim = embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_attention_layers)])
        self.fnn_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_attention_layers)])
        self.fnn_layers = nn.ModuleList([nn.Linear(embed_dim, fnn_dim) for _ in range(num_attention_layers)])  # Fully connected layers
        self.fnn_out_layers = nn.ModuleList([nn.Linear(fnn_dim, embed_dim) for _ in range(num_attention_layers - 1)])
        self.fnn_last_layer = nn.Linear(fnn_dim, 1)
        self.to(device)

    def forward(self, state):
        for i in range(len(self.attention_layers)):
            attention = self.attention_layer_norm[i](state)   # Pre-Layer Norm
            query = attention.clone()
            key = attention.clone()
            value = attention.clone()
            attention = state + self.attention_layers[i](query, key, value)[0]
            fnn = self.fnn_layer_norm[i](attention)     # Pre-Layer Norm
            fnn = F.elu(self.fnn_layers[i](fnn))
            if i < len(self.attention_layers) - 1:
                state = F.elu(attention + self.fnn_out_layers[i](fnn))
            else:
                q_value_of_options = self.fnn_last_layer(fnn)
        q_value_of_options = q_value_of_options.view(state.shape[0],state.shape[1])
        return q_value_of_options
    
class Q_Critic_NN(nn.Module):
    """
    Args
    -------
    input_dim:
        = num_components * embed_dim
    output_dim:
        = num_components
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, device='cpu'):
        super(Q_Critic_NN, self).__init__()
        self.device = device
        self.input_dim = input_dim 
        hidden_dim = 10*input_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.to(device)

    def forward(self, state):
        """
        Args
        -------
        state:
            Shape is like (num_batch, num_components, embed_dim)
        
        Returns
        -------
        x:
            Shape is like (num_batch, num_components)
        """
        x = state.float().view(-1, self.input_dim)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Actor_Attention(nn.Module):
    """
    Description
    -------
    This is decoder-only. The `q` for the decoder is the option embeddings.

    Returns
    -------
    action_probs:
        The dimension is the max number of options of each component. We assume that we have picked up the component index in the Critic.
    """
    def __init__(self, embed_dim, output_dim, num_attention_layers, num_heads, fnn_dim=None, device='cpu'):
        super(Actor_Attention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        fnn_dim = 1*embed_dim if fnn_dim is None else fnn_dim
        self.output_dim = output_dim
        self.attention_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_attention_layers)])
        self.attention_layers = nn.ModuleList([MultiheadAttention(embed_dim = embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_attention_layers)])
        self.fnn_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_attention_layers)])
        self.fnn_layers = nn.ModuleList([nn.Linear(embed_dim, fnn_dim) for _ in range(num_attention_layers)])  # Fully connected layers
        self.fnn_out_layers = nn.ModuleList([nn.Linear(fnn_dim, embed_dim) for _ in range(num_attention_layers - 1)])
        self.fnn_last_layer = nn.Linear(fnn_dim, output_dim)
        self.to(device)

    def forward(self, state, option, action_mask):
        q = option.view(state.shape[0], -1, state.shape[2]) * 8 
        for i in range(len(self.attention_layers)):
            k = torch.concat([q.clone(), state], dim=1)
            v = torch.concat([q.clone(), state], dim=1)
            q_norm = self.attention_layer_norm[i](q)   # Pre-Layer Norm
            k_norm = self.attention_layer_norm[i](k)
            v_norm = self.attention_layer_norm[i](v)
            # attention = q + self.attention_layers[i](q_norm, k_norm, v_norm)[0]
            attention = self.attention_layers[i](q_norm, k_norm, v_norm)[0]
            fnn = self.fnn_layer_norm[i](attention)     # Pre-Layer Norm
            fnn = F.elu(self.fnn_layers[i](fnn))
            if i < len(self.attention_layers) - 1:
                # q = F.elu(attention + self.fnn_out_layers[i](fnn))
                q = self.fnn_out_layers[i](fnn)
            else:
                state = self.fnn_last_layer(fnn)
        action_probs = state.view(state.shape[0], self.output_dim)
        # `action_mask` is indicated by "True" or "False". So we add a very small number to the elements being masked to block out gradient propagation.
        action_probs = action_probs + ~torch.tensor(action_mask).to(self.device)*(-1e9)
        action_probs = F.softmax(action_probs, dim=1)
        action_probs = action_probs + torch.tensor(action_mask).to(self.device)*0.0001 # We must handle the case that sometimes all outputs are zeros.
        return action_probs
    
class Actor_Transformer(nn.Module):
    def __init__(self, state_embed_dim, option_embed_dim, output_dim, num_encoder_layers, num_decoder_layers, num_heads, fnn_dim=None, device='cpu'):
        super(Actor_Transformer, self).__init__()
        self.device = device
        self.state_embed_dim = state_embed_dim
        self.option_embed_dim = option_embed_dim
        encoder_fnn_dim = 2*state_embed_dim if fnn_dim is None else fnn_dim
        decoder_fnn_dim = 2*option_embed_dim if fnn_dim is None else fnn_dim
        self.output_dim = output_dim
        self.encoder_layer_norm = nn.LayerNorm(state_embed_dim)
        self.decoder_layer_norm = nn.LayerNorm(option_embed_dim)

        self.encoder_att_layers = nn.ModuleList([MultiheadAttention(embed_dim = state_embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_encoder_layers)])
        self.encoder_fnn_layers = nn.ModuleList([nn.Linear(state_embed_dim, encoder_fnn_dim) for _ in range(num_encoder_layers)])  # Fully nueral net
        self.encoder_fnn_out_layers = nn.ModuleList([nn.Linear(encoder_fnn_dim, state_embed_dim) for _ in range(num_encoder_layers - 1)])
        self.encoder_fc_last_layer = nn.Linear(encoder_fnn_dim, option_embed_dim)

        self.decoder_att_layers = nn.ModuleList([MultiheadAttention(embed_dim = option_embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_decoder_layers)])
        self.decoder_fnn_layers = nn.ModuleList([nn.Linear(option_embed_dim, decoder_fnn_dim) for _ in range(num_decoder_layers)])  # Fully nueral net
        self.decoder_fnn_out_layers = nn.ModuleList([nn.Linear(decoder_fnn_dim, option_embed_dim) for _ in range(num_decoder_layers - 1)])
        self.fnn_last_layer = nn.Linear(decoder_fnn_dim, output_dim)
        self.to(device)
    
    def forward(self, state, option, action_mask):
        option = option.view(state.shape[0], -1, option.shape[2]) * 8
        # Encode
        for i in range(len(self.encoder_att_layers)):
            q_encoder_norm = self.encoder_layer_norm(state)   # Pre-Layer Norm
            k_encoder_norm = self.encoder_layer_norm(state)
            v_encoder_norm = self.encoder_layer_norm(state)
            state = state + self.encoder_att_layers[i](q_encoder_norm, k_encoder_norm, v_encoder_norm)[0]
            q_encoder_norm = self.encoder_layer_norm(state)   # Pre-Layer Norm
            q_encoder = F.elu(self.encoder_fnn_layers[i](q_encoder_norm))
            if i < len(self.encoder_att_layers) - 1:
                state = state + self.encoder_fnn_out_layers[i](q_encoder)
            else:
                state = self.encoder_fc_last_layer(q_encoder)
        state_norm = self.decoder_layer_norm(state)
        
        # Decode
        for i in range(len(self.decoder_att_layers)):
            q_decoder_norm = self.decoder_layer_norm(option)
            # k_decoder = torch.concat([option.clone(), state], dim=1)
            # v_decoder = torch.concat([option.clone(), state], dim=1)
            k_decoder_norm = state_norm.clone()
            v_decoder_norm = state_norm.clone()
            # k_decoder_norm = self.decoder_layer_norm(k_decoder)
            # v_decoder_norm = self.decoder_layer_norm(v_decoder)
            option = option + self.decoder_att_layers[i](q_decoder_norm, k_decoder_norm, v_decoder_norm)[0]
            fnn = self.decoder_layer_norm(option)     # Pre-Layer Norm
            fnn = F.elu(self.decoder_fnn_layers[i](fnn))
            if i < len(self.decoder_att_layers) - 1:
                option = F.elu(option + self.decoder_fnn_out_layers[i](fnn))
            else:
                state = self.fnn_last_layer(fnn)
        action_probs = state.view(state.shape[0], self.output_dim)
        # `action_mask` is indicated by "True" or "False". So we add a very small number to the elements being masked to block out gradient propagation.
        action_probs = action_probs + ~torch.tensor(action_mask).to(self.device)*(-1e9)
        action_probs = F.softmax(action_probs, dim=1)
        action_probs = action_probs + torch.tensor(action_mask).to(self.device)*0.0001 # We must handle the case that sometimes all outputs are zeros.
        return action_probs
        
    
class V_Critic(nn.Module):
    def __init__(self, state_dim, device='cpu'):
        super(V_Critic, self).__init__()
        self.device = device
        fc_dim = 10*state_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, 1)
        self.to(device)
    
    def forward(self, state):
        x = F.elu(self.fc1(state.float()))
        x = F.elu(self.fc2(x))
        value = self.fc3(x)
        return value

class Actor_NN(nn.Module):
    def __init__(self, state_dim, output_dim, device='cpu'):
        super(Actor_NN, self).__init__()
        self.device = device
        self.input_dim = state_dim
        fc_dim = 10*state_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.to(device)

    def forward(self, state, option, action_mask):
        """
        Args
        -------
        option:
            Should be the idx of the specified component.
        """
        x = torch.concat([state, 2*option], dim=1).float()
        x = x.view(-1, self.input_dim)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        action_probs = self.fc3(x)
        action_probs = action_probs + ~torch.tensor(action_mask).to(self.device)*(-1e9)
        action_probs = F.softmax(action_probs, dim=1)
        action_probs = action_probs + torch.tensor(action_mask).to(self.device)*0.0001
        return action_probs
    
class Actor_NN_no_option(nn.Module):
    def __init__(self, state_dim, output_dim, device='cpu'):
        super(Actor_NN_no_option, self).__init__()
        self.device = device
        self.output_dim = torch.tensor(output_dim)
        fc_dim = 4*state_dim
        self.fc1 = nn.Linear(state_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.to(device)

    def forward(self, state, action_mask):
        x = F.elu(self.fc1(state.float()))
        x = F.elu(self.fc2(x))
        action_probs = self.fc3(x)
        action_probs = action_probs + ~action_mask*(-1e9)
        action_probs = F.softmax(action_probs/torch.sqrt(self.output_dim), dim=1)
        action_probs = action_probs + action_mask*0.0001
        return action_probs