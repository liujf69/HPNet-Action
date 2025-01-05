import torch
import torch.nn as nn
import torch.nn.functional as F
 
class FiLM(nn.Module):
    def __init__(self, condition_dim: int = 256, proj_dim: int = 512):
        super(FiLM, self).__init__()
        self.fc_gamma = nn.Sequential(
            nn.Linear(condition_dim, proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(condition_dim, proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
        gamma = self.fc_gamma(skeleton).unsqueeze(1) # B 1 proj_dim
        beta = self.fc_beta(skeleton).unsqueeze(1) # B 1 proj_dim
        output = gamma * text + beta # B Num_labels proj_dim
        return output
    
class TRMM(nn.Module):
    def __init__(self, condition_dim: int = 256, proj_dim: int = 512):
        super(TRMM, self).__init__()
        self.fc_gamma = nn.Sequential(
            nn.Linear(condition_dim, proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(condition_dim, proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace = True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace = True)
        )
        self.tanh = nn.Tanh()
        
    def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
        bs, num_labels, proj_dim = text.shape
        gamma = self.fc_gamma(skeleton).unsqueeze(1) # B 1 proj_dim
        beta = self.fc_beta(skeleton).unsqueeze(1) # B 1 proj_dim
        ske_film_text = gamma * text + beta # B Num_labels proj_dim
        text_m1 = self.fc1(text).mean(-2) # B Num_labels proj_dim -> B 1 proj_dim
        text_m2 = self.fc2(text).mean(-2) # B Num_labels proj_dim -> B 1 proj_dim
        text_m = self.tanh(text_m1.unsqueeze(-1) - text_m2.unsqueeze(-2)) # B 1 proj_dim proj_dim
        text_sub = torch.einsum('bnd,bud->bnu', text, text_m) # B Num_labels 1 proj_dim -> B Num_labels proj_dim
        text = text_sub + ske_film_text
        return text

# class FiLM3(nn.Module):
#     def __init__(self, condition_dim: int = 256, proj_dim: int = 512):
#         super(FiLM3, self).__init__()
#         self.fc_gamma = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.fc_beta = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.tanh = nn.Tanh()
        
#     def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         bs, num_labels, proj_dim = text.shape
#         gamma = self.fc_gamma(skeleton).unsqueeze(1) # B 1 proj_dim
#         beta = self.fc_beta(skeleton).unsqueeze(1) # B 1 proj_dim
#         ske_film_text = gamma * text + beta # B Num_labels proj_dim
#         text_m1 = self.fc1(ske_film_text).mean(-2) # B Num_labels proj_dim -> B 1 proj_dim
#         text_m2 = self.fc2(ske_film_text).mean(-2) # B Num_labels proj_dim -> B 1 proj_dim
#         text_m = self.tanh(text_m1.unsqueeze(-1) - text_m2.unsqueeze(-2)) # B 1 proj_dim proj_dim
#         text_sub = torch.einsum('bnd,bud->bnu', text, text_m) # B Num_labels 1 proj_dim -> B Num_labels proj_dim
#         return text_sub
    
# class FiLM4(nn.Module):
#     def __init__(self, condition_dim: int = 256, proj_dim: int = 512):
#         super(FiLM4, self).__init__()
#         self.fc_gamma = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.fc_beta = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.tanh = nn.Tanh()
        
#     def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         bs, num_labels, proj_dim = text.shape
#         gamma = self.fc_gamma(skeleton).unsqueeze(1) # B 1 proj_dim
#         beta = self.fc_beta(skeleton).unsqueeze(1) # B 1 proj_dim
#         ske_film_text = gamma * text + beta # B Num_labels proj_dim
#         text_m1 = self.fc1(ske_film_text) # B Num_labels proj_dim
#         text_m2 = self.fc2(ske_film_text) # B Num_labels proj_dim
#         text_m = self.tanh(text_m1.unsqueeze(-1) - text_m2.unsqueeze(-2)) # B Num_labels proj_dim proj_dim
#         text_sub = torch.einsum('blnd,blud->blnu', text.unsqueeze(-2), text_m).squeeze(-2) # B Num_labels 1 proj_dim -> B Num_labels proj_dim
#         return text_sub
    
# class Video_FiLM(nn.Module):
#     def __init__(self, condition_dim: int = 512, proj_dim: int = 512):
#         super(Video_FiLM, self).__init__()
#         self.fc_gamma = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.fc_beta = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
        
#     def forward(self, text: torch.Tensor, video: torch.Tensor): # text: [B, Num_labels, proj_dim] video: [B, condition_dim]
#         gamma = self.fc_gamma(video).unsqueeze(1) # B 1 proj_dim
#         beta = self.fc_beta(video).unsqueeze(1) # B 1 proj_dim
#         output = gamma * text + beta # B Num_labels proj_dim
#         return output

# class Text_FiLM(nn.Module):
#     def __init__(self, condition_dim: int = 512, proj_dim: int = 256):
#         super(Text_FiLM, self).__init__()
#         self.fc_gamma = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.fc_beta = nn.Sequential(
#             nn.Linear(condition_dim, proj_dim),
#             nn.Linear(proj_dim, proj_dim)
#         )
        
#     def forward(self, skeleton: torch.Tensor, text: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         gamma = self.fc_gamma(text).mean(1) # B 1 proj_dim
#         beta = self.fc_beta(text).mean(1) # B 1 proj_dim
#         output = gamma * skeleton + beta # B 1 proj_dim
#         return output

# class Bi_Directional_FiNlM(nn.Module):
#     def __init__(self, ske_dim: int = 256, text_dim: int = 512):
#         super(Bi_Directional_FiNlM, self).__init__()
#         self.fc_gamma_1 = nn.Sequential(
#             nn.Linear(text_dim, ske_dim),
#             nn.Linear(ske_dim, ske_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.fc_beta_1 = nn.Sequential(
#             nn.Linear(text_dim, ske_dim),
#             nn.Linear(ske_dim, ske_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.alpha_1 = nn.Parameter(torch.tensor(1.0))
        
#         self.fc_gamma_2 = nn.Sequential(
#             nn.Linear(ske_dim, text_dim),
#             nn.Linear(text_dim, text_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.fc_beta_2 = nn.Sequential(
#             nn.Linear(ske_dim, text_dim),
#             nn.Linear(text_dim, text_dim),
#             nn.ReLU(inplace = True)
#         )
        
#     def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         text_gamma = self.fc_gamma_1(text).mean(1) # B 1 proj_dim
#         text_beta = self.fc_beta_1(text).mean(1) # B 1 proj_dim
#         skeleton = skeleton + self.alpha_1*(text_gamma * skeleton + text_beta) # B proj_dim
        
#         ske_gamma = self.fc_gamma_2(skeleton.unsqueeze(1)) # B 1 proj_dim
#         ske_beta = self.fc_beta_2(skeleton.unsqueeze(1)) # B 1 proj_dim
#         text = ske_gamma * text + ske_beta # B Num_labels proj_dim
#         return text

# class Bi_Directional_FiNlM(nn.Module):
#     def __init__(self, ske_dim: int = 256, text_dim: int = 512):
#         super(Bi_Directional_FiNlM, self).__init__()
#         self.fc_gamma_1 = nn.Sequential(
#             nn.Linear(text_dim, ske_dim),
#             nn.Linear(ske_dim, ske_dim)
#         )
#         self.fc_beta_1 = nn.Sequential(
#             nn.Linear(text_dim, ske_dim),
#             nn.Linear(ske_dim, ske_dim),
#         )
#         self.N_linear_func_1 = nn.ReLU(inplace = True)
        
#         self.fc_gamma_2 = nn.Sequential(
#             nn.Linear(ske_dim, text_dim),
#             nn.Linear(text_dim, text_dim),
#         )
#         self.fc_beta_2 = nn.Sequential(
#             nn.Linear(ske_dim, text_dim),
#             nn.Linear(text_dim, text_dim),
#         )
#         self.N_linear_func_2 = nn.ReLU(inplace = True)
        
#     def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         text_gamma = self.fc_gamma_1(text).mean(1) # B 1 proj_dim
#         text_beta = self.fc_beta_1(text).mean(1) # B 1 proj_dim
#         skeleton = skeleton + self.N_linear_func_1(text_gamma * skeleton + text_beta) # B proj_dim
        
#         ske_gamma = self.fc_gamma_2(skeleton.unsqueeze(1)) # B 1 proj_dim
#         ske_beta = self.fc_beta_2(skeleton.unsqueeze(1)) # B 1 proj_dim
#         text = self.N_linear_func_2(ske_gamma * text + ske_beta) # B Num_labels proj_dim
#         return text

# class Bi_Directional_FiNlM(nn.Module):
#     def __init__(self, ske_dim: int = 256, text_dim: int = 512):
#         super(Bi_Directional_FiNlM, self).__init__()
#         self.fc_gamma_1 = nn.Sequential(
#             nn.Linear(text_dim, ske_dim),
#             nn.Linear(ske_dim, ske_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.fc_beta_1 = nn.Sequential(
#             nn.Linear(text_dim, ske_dim),
#             nn.Linear(ske_dim, ske_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.N_linear_func_1 = nn.ReLU(inplace = True)
#         self.alpha_1 = nn.Parameter(torch.tensor(1.0))
        
#         self.fc_gamma_2 = nn.Sequential(
#             nn.Linear(ske_dim, text_dim),
#             nn.Linear(text_dim, text_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.fc_beta_2 = nn.Sequential(
#             nn.Linear(ske_dim, text_dim),
#             nn.Linear(text_dim, text_dim),
#             nn.ReLU(inplace = True)
#         )
#         self.N_linear_func_2 = nn.ReLU(inplace = True)
        
#     def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         text_gamma = self.fc_gamma_1(text) # B Num_labels proj_dim
#         text_beta = self.fc_beta_1(text) # B Num_labels proj_dim
#         skeleton = skeleton.unsqueeze(1)
#         skeleton = skeleton + self.alpha_1 * self.N_linear_func_1(text_gamma * skeleton + text_beta) # B Num_labels proj_dim
        
#         ske_gamma = self.fc_gamma_2(skeleton) # B Num_labels proj_dim
#         ske_beta = self.fc_beta_2(skeleton) # B Num_labels proj_dim
#         text = self.N_linear_func_2(ske_gamma * text + ske_beta) # B Num_labels proj_dim
#         return text

# class MOE_FiLM(nn.Module):
#     def __init__(self, ske_dim: int = 256, text_dim: int = 512, num_experts: int = 2):
#         super(MOE_FiLM, self).__init__()
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList([FiLM(condition_dim = ske_dim, proj_dim = text_dim) for _ in range(self.num_experts)])
#         self.gating_network = nn.Linear(text_dim, num_experts)
        
#     def forward(self, text: torch.Tensor, skeleton: torch.Tensor): # text: [B, Num_labels, proj_dim] skeleton: [B, condition_dim]
#         B, Num_labels, _ = text.shape
#         # [B, Num_labels, Num_experts, proj_dim] -> [B*Num_labels, Num_experts, proj_dim]
#         expert_outputs = torch.stack([expert(text, skeleton) for expert in self.experts], dim = -2).reshape(B*Num_labels, self.num_experts, -1)
#         # [B, Num_labels, num_experts] -> [B*Num_labels, Num_experts]
#         gating_scores = F.softmax(self.gating_network(text), dim = -1).reshape(B*Num_labels, self.num_experts)
#         text = torch.bmm(gating_scores.unsqueeze(1), expert_outputs).squeeze(1).reshape(B, Num_labels, -1) # [B, Num_labels, proj_dim]
#         return text # [B, Num_labels, proj_dim]
 
# if __name__ == "__main__":
#     B = 4
#     Num_labels = 120
#     D = 512
#     text_features = torch.rand(B, Num_labels, D)
    
#     ske_D = 256
#     skeleton_features =  torch.rand(B, 256)
    
#     # demo = FiLM(ske_D, D)
#     # output = demo(text_features, skeleton_features)
    
#     # text_film_demo = Text_FiLM(D, ske_D)
#     # output = text_film_demo(skeleton_features, text_features)
    
#     # demo = Bi_Directional_FiNlM(ske_D, D)
#     # output = demo(text_features, skeleton_features)
    
#     # demo = MOE_FiLM(ske_D, D)
#     # output = demo(text_features, skeleton_features)
    
#     demo = FiLM4(ske_D, D)
#     output = demo(text_features, skeleton_features)
    
#     print("All Done!")