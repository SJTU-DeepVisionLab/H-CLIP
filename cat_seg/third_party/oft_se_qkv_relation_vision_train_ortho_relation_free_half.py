import clip.model
import torch.nn as nn
import clip
from torch.nn import functional as F
import torch
from einops import rearrange
from . import model_oft_relation_both_free_half
#VisualTransformer
#from ..cat_seg_model import CATSeg

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def init_weights_eye(m):
	if type(m) == nn.Linear:
		nn.init.eye_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)



def oft_forward(self, x):
    B, N, C = x.shape
    res_x = x
    orig_dtype = x.dtype

    _, N1, _, _ = self.attn_q_proj_oft_layer_R.shape
    attn_tensor = rearrange(self.attn_q_proj_oft_layer_R.cuda().to(orig_dtype), 'B1 N1 L1 M1 -> B1 (N1 L1) M1')
    attn_tensor = self.attn_q_proj_oft_relation_m_R[int((self.count-4)/4)](attn_tensor)[:,:, :4]
    attn_tensor = rearrange(attn_tensor, 'B1 (N1 L1) M1 -> M1 B1 N1 L1', N1=N1)
    attn_re = self.attn_q_proj_oft_relation_l_R(attn_tensor)[..., self.count-4:self.count]
    q_R, k_R, v_R, proj_R = attn_re[..., 0], attn_re[..., 1], attn_re[..., 2], attn_re[..., 3]

    qkv = self.to_qkv_oft(self.attn.in_proj_weight, self.attn.in_proj_bias,self.ln_1(x), q_R, k_R, v_R)
    qkv = qkv.reshape(B,N,3,
                    self.attn.num_heads,
                    C // self.attn.num_heads).permute(
                    2, 1, 3, 0, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2,-1)) * (float(self.attn.head_dim) ** -0.5)
    attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(1).cuda().to(orig_dtype)
    attn = attn.softmax(dim=-1)
    oft_out = self.to_out_oft(self.attn.out_proj.weight, self.attn.out_proj.bias,((attn @ v).transpose(1,2)).permute(1,0,2,3).reshape(B, N, C), proj_R)
    oft_out = self.dp(oft_out)
    final = res_x + oft_out #+ ori_attn_x
    #final = res_x + oft_out
    final = final + self.mlp(self.ln_2(final))
    return final



def set_oft(model, dim=8, hidden_size=512, length=12, s=0.1, r=4, count=0):
    for _ in model.children():
        #print('length',length)
        if isinstance(_, model_oft_relation_both_free_half.ResidualAttentionBlock):
            count+=4
            _.to_qkv_oft = OFTLinearLayer_both_relation_in_text(hidden_size,hidden_size,r=r)
            _.to_out_oft = OFTLinearLayer_both_relation_text(hidden_size,hidden_size,r=r)
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            _.dim = dim
            _.hidden_size = hidden_size
            _.count = count
            bound_method = oft_forward.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_oft(_, dim, hidden_size, length, s, r, count)
    #print('count',count)


def oft_forward_vision(self, x):
    orig_dtype = x.dtype
    #calculate(self.attn.in_proj_weight)
    _, N1, _, _ = self.attn_q_proj_oft_layer_R.shape
    attn_tensor = rearrange(self.attn_q_proj_oft_layer_R.cuda().to(orig_dtype), 'B1 N1 L1 M1 -> B1 (N1 L1) M1')
    attn_tensor = self.attn_q_proj_oft_relation_m_R[int((self.count-4)/4)](attn_tensor)[:,:,-6:]
    attn_tensor = rearrange(attn_tensor, 'B1 (N1 L1) M1 -> M1 B1 N1 L1', N1=N1)
    attn_re = self.attn_q_proj_oft_relation_l_R(attn_tensor)[..., self.count-4:self.count]
    q_R, k_R, v_R, proj_R = attn_re[..., 0], attn_re[..., 1], attn_re[..., 2], attn_re[..., 3]

    if self.count <= 44:
        B, N, C = x.shape
        res_x = x
        qkv = self.to_qkv_oft_vision(self.attn.in_proj_weight, self.attn.in_proj_bias,self.ln_1(x), q_R, k_R, v_R)
        qkv = qkv.reshape(B,N,3,
                        self.n_head,
                        C // self.n_head).permute(
                        2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * (float(self.attn.head_dim) ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = ((attn @ v).transpose(1,2)).permute(1,0,2,3).reshape(B, N, C)
        oft_out = self.to_out_oft_vision(self.attn.out_proj.weight, self.attn.out_proj.bias, attn, proj_R)
        oft_out = self.dp(oft_out)
        final = res_x + oft_out 
        final = final + self.mlp(self.ln_2(final))
        return final
    else:
        y = self.to_qkv_oft_vision(self.attn.in_proj_weight, self.attn.in_proj_bias,self.ln_1(x), q_R, k_R, v_R)
        L, N, D = y.shape # L N 3D        
        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0, 3).reshape(3 * N, L, D // 3)
        y = self.to_out_oft_vision(self.attn.out_proj.weight, self.attn.out_proj.bias, y, proj_R)        
        q, k, v = y.tensor_split(3, dim=0)      
        v = v.transpose(1, 0) + x[:1] # L N D
        v = v + self.mlp(self.ln_2(v))
        return v

def set_oft_vision(model, dim=8, hidden_size=512, length=12, s=0.1, r=6, count=0):

    for _ in model.children():
        #print('length',length)
        if isinstance(_, model_oft_relation_both_free_half.ResidualAttentionBlock):
            count+=4
            print('_.count',count)
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            _.dim = dim
            
            if count <=0:
                _.to_qkv_oft_vision = OFTLinearLayer_both_relation_in_text(hidden_size,hidden_size,r=r)
                _.to_out_oft_vision = OFTLinearLayer_both_relation_text(hidden_size,hidden_size,r=r)
            else:
                _.to_qkv_oft_vision = OFTLinearLayer_both_relation_in(hidden_size,hidden_size,r=r)
                _.to_out_oft_vision = OFTLinearLayer_both_relation(hidden_size,hidden_size,r=r)

            _.hidden_size = hidden_size
            _.count = count
            bound_method = oft_forward_vision.__get__(_, _.__class__)
            if count <= 44:
                setattr(_, 'forward', bound_method)
            else:
                setattr(_, 'forward_dense', bound_method)
        elif len(list(_.children())) != 0:
            set_oft_vision(_, dim, hidden_size, length, s, r, count)
    print('count',count)



class OFTLinearLayer_both_relation_in(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=8, is_coft=False):
        super(OFTLinearLayer_both_relation_in, self).__init__()

        # Define the reduction rate:
        self.r = r
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share

        #self.shortcut = nn.Parameter(torch.tensor([0.]))

    def forward(self, attn, bias, x, q_R = None, k_R = None, v_R = None):
        orig_dtype = x.dtype
        dtype = q_R.dtype
        #print(self.R.sum())
        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    q_R.copy_(project(q_R, eps=self.eps))
                    k_R.copy_(project(k_R, eps=self.eps))
                    v_R.copy_(project(v_R, eps=self.eps))
            # orth_rotate_q = self.cayley(q_R)
            # orth_rotate_k = self.cayley(k_R)
            # orth_rotate_v = self.cayley(v_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    q_R.copy_(project(q_R, eps=self.eps))
                    k_R.copy_(project(k_R, eps=self.eps))
                    v_R.copy_(project(v_R, eps=self.eps))
            #orth_rotate_q = self.cayley_batch(q_R)
            #orth_rotate_k = self.cayley_batch(k_R)
            #orth_rotate_v = self.cayley_batch(v_R)

        # Block-diagonal parametrization
        block_diagonal_matrix_q = self.block_diagonal(q_R)
        block_diagonal_matrix_k = self.block_diagonal(k_R)
        block_diagonal_matrix_v = self.block_diagonal(v_R)

        # fix filter
        fix_filt = attn.data
        q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=0)
        #fix_filt = torch.transpose(fix_filt, 0, 1)
        filt_q = torch.mm(q_proj_weight.to(orig_dtype), block_diagonal_matrix_q.to(orig_dtype))
        #if q_re is not None:
            #filt_q = filt_q * q_re   
        filt_k = torch.mm(k_proj_weight.to(orig_dtype), block_diagonal_matrix_k.to(orig_dtype))
        #if k_re is not None:
            #filt_k = filt_k * k_re   
        filt_v = torch.mm(v_proj_weight.to(orig_dtype), block_diagonal_matrix_v.to(orig_dtype))
        #if v_re is not None:

        # if q_re is not None and k_re is not None and v_re is not None:
        #     filt_q = filt_q * q_re#0.5 * (filt_q + q_proj_weight * q_re)

        #     filt_k = filt_k * k_re#0.5 * (filt_k + k_proj_weight * k_re)

        #     filt_v = filt_v * v_re # 0.5 * (filt_v + v_proj_weight * v_re)
            
        filt  = torch.cat([filt_q, filt_k, filt_v], dim=0)


        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out
    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

class OFTLinearLayer_both_relation_in_text(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=8, is_coft=False):
        super(OFTLinearLayer_both_relation_in_text, self).__init__()

        # Define the reduction rate:
        self.r = r
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share

        #self.shortcut = nn.Parameter(torch.tensor([0.]))

    def forward(self, attn, bias, x,  q_R = None, k_R = None, v_R = None):
        orig_dtype = x.dtype
        dtype = q_R.dtype
        #print(self.R.sum())
        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    q_R.copy_(project(q_R, eps=self.eps))
                    k_R.copy_(project(k_R, eps=self.eps))
                    v_R.copy_(project(v_R, eps=self.eps))
            orth_rotate_q = self.cayley(q_R)
            orth_rotate_k = self.cayley(k_R)
            orth_rotate_v = self.cayley(v_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    q_R.copy_(project(q_R, eps=self.eps))
                    k_R.copy_(project(k_R, eps=self.eps))
                    v_R.copy_(project(v_R, eps=self.eps))
            orth_rotate_q = self.cayley_batch(q_R)
            orth_rotate_k = self.cayley_batch(k_R)
            orth_rotate_v = self.cayley_batch(v_R)

        # Block-diagonal parametrization
        block_diagonal_matrix_q = self.block_diagonal(orth_rotate_q)
        block_diagonal_matrix_k = self.block_diagonal(orth_rotate_k)
        block_diagonal_matrix_v = self.block_diagonal(orth_rotate_v)

        # fix filter
        fix_filt = attn.data
        q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=0)
        #fix_filt = torch.transpose(fix_filt, 0, 1)
        filt_q = torch.mm(q_proj_weight.to(orig_dtype), block_diagonal_matrix_q.to(orig_dtype))
        #if q_re is not None:
            #filt_q = filt_q * q_re   
        filt_k = torch.mm(k_proj_weight.to(orig_dtype), block_diagonal_matrix_k.to(orig_dtype))
        #if k_re is not None:
            #filt_k = filt_k * k_re   
        filt_v = torch.mm(v_proj_weight.to(orig_dtype), block_diagonal_matrix_v.to(orig_dtype))
        #if v_re is not None:

        # if q_re is not None and k_re is not None and v_re is not None:
        #     filt_q = filt_q * q_re#0.5 * (filt_q + q_proj_weight * q_re)

        #     filt_k = filt_k * k_re#0.5 * (filt_k + k_proj_weight * k_re)

        #     filt_v = filt_v * v_re # 0.5 * (filt_v + v_proj_weight * v_re)
            
        filt  = torch.cat([filt_q, filt_k, filt_v], dim=0)


        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out
    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))


class OFTLinearLayer_both_relation(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=8, is_coft=False):
        super(OFTLinearLayer_both_relation, self).__init__()

        # Define the reduction rate:
        self.r = r
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share
        # Define the trainable matrix parameter: R


    def forward(self, attn, bias, x, proj_R = None):
        orig_dtype = x.dtype
        dtype = proj_R.dtype
        #print(self.R.sum())
        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    proj_R.copy_(project(proj_R, eps=self.eps))
            #orth_rotate = self.cayley(proj_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    proj_R.copy_(project_batch(proj_R, eps=self.eps))
            #orth_rotate = self.cayley_batch(proj_R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(proj_R)

        # fix filter
        fix_filt = attn.data
        #fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix.to(orig_dtype), fix_filt.to(orig_dtype))
        #filt = torch.transpose(filt, 0, 1)

        # if proj_re is not None:
        #     filt = filt * proj_re #0.5 * (filt + fix_filt.to(dtype) * proj_re)

            #filt = 0.5*filt + 0.5*filt_re
        # Apply the trainable identity matrix
        # bias_term = attn.bias.data if attn.bias is not None else None
        # if bias_term is not None:
        #     bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

class OFTLinearLayer_both_relation_text(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_share=False, eps=6e-5, r=8, is_coft=False):
        super(OFTLinearLayer_both_relation_text, self).__init__()

        # Define the reduction rate:
        self.r = r
        
        # Check whether to use the constrained variant COFT 
        self.is_coft = is_coft

        assert in_features % self.r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        # self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        # self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))
        
        # Define the fixed Linear layer: v
        # self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        self.block_share = block_share
        # Define the trainable matrix parameter: R


    def forward(self, attn, bias, x,proj_R = None):
        orig_dtype = x.dtype
        dtype = proj_R.dtype
        #print(self.R.sum())
        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    proj_R.copy_(project(proj_R, eps=self.eps))
            orth_rotate = self.cayley(proj_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    proj_R.copy_(project_batch(proj_R, eps=self.eps))
            orth_rotate = self.cayley_batch(proj_R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = attn.data
        #fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix.to(orig_dtype), fix_filt.to(orig_dtype))
        #filt = torch.transpose(filt, 0, 1)

        # if proj_re is not None:
        #     filt = filt * proj_re #0.5 * (filt + fix_filt.to(dtype) * proj_re)

            #filt = 0.5*filt + 0.5*filt_re
        # Apply the trainable identity matrix
        # bias_term = attn.bias.data if attn.bias is not None else None
        # if bias_term is not None:
        #     bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I - skew, torch.inverse(I + skew))

        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

def project_vision(R, eps):
    I = torch.eye((R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch_vision(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.eye((R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

def is_orthogonal(R, eps=1e-5):
    with torch.no_grad():
        RtR = torch.matmul(R.t(), R)
        diff = RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device)
        if (torch.any(diff > eps)):
            R = R - torch.matmul(R, diff)
            #print('gotta you')
        return R 

def is_batch_orthogonal(R, eps=1e-5):
    with torch.no_grad():
        RtR = torch.matmul(R.transpose(1, 2), R)
        diff = RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device)
        if (torch.any(diff > eps)): 
            R = R - torch.matmul(R, diff)
            #print(R)
        return R



def calculate(attn):
    fix_filt = attn.data
    q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=1)
    dist_matrix = torch.cdist(q_proj_weight, q_proj_weight, p=2).to(float)  
    torch.diagonal(dist_matrix).fill_(1e8)
    inv_dist_matrix = 1.0 / dist_matrix
    sum_inv_distances_q = inv_dist_matrix.sum()
    print(sum_inv_distances_q)  

    dist_matrix = torch.cdist(v_proj_weight, v_proj_weight, p=2).to(float)  
    torch.diagonal(dist_matrix).fill_(1e8)
    inv_dist_matrix = 1.0 / dist_matrix
    sum_inv_distances_v = inv_dist_matrix.sum()
    print(sum_inv_distances_v)    

    return