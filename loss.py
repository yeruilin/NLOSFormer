import torch.nn as nn
import torch
import torch.nn.functional as F

def TVLoss(x):
    # Calculate image height and width
    h_x, w_x = x.size()[2], x.size()[3]
    
    # Compute gradients in horizontal and vertical directions
    diff_h = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
    
    # Calculate sum of squared gradients
    tv_h = diff_h.pow(2).sum([1, 2, 3])
    tv_w = diff_w.pow(2).sum([1, 2, 3])

    # Compute total variation loss
    tv = (tv_h + tv_w).mean()
    return tv

def L1SmoothLoss(x):
    l1 = torch.mean(torch.abs(x))
    return l1

def L1Loss(x, gt):
    return torch.mean(torch.abs(x - gt))

def MSELoss(x, gt):
    mse = torch.mean((x - gt) ** 2)  # Using mean squared error
    return mse

def CrossEntropyLoss(y, y_pred):  # Requires y and y_pred after softmax
    loss = torch.mean(torch.sum(-y * torch.log(y_pred), dim=1))
    return loss

# Custom loss function class
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    
    def forward(self, pred_img, gt_img, pred_kernel, reduced_kernel):
        coeff = 1e-3  # Use 1e-5 for MSE, 1e-4 for L1

        mse1 = L1Loss(pred_img, gt_img)  # MSE performs worse than L1
        smoothl1 = coeff * TVLoss(pred_img)
        mse2 = L1Loss(pred_kernel, reduced_kernel)
        loss=mse1+mse2
        
        return loss

class RestoreLoss(nn.Module):
    def __init__(self):
        super(RestoreLoss, self).__init__()
    
    def forward(self, pred_img,gt_img):
        # B,C,H,W=pred_kernel.shape
        mse=L1Loss(pred_img,gt_img)
        loss=mse
        return loss

# ==============================================================================
# MS Loss: multi-scale loss for LMS-NLOS
# ==============================================================================
class MSLoss(nn.Module):
    """
    Multi-Scale Loss function from the paper.
    Combines L1 content loss and FFT-based auxiliary loss.
    Eq. 8, 9, 10 in the paper.
    """
    def __init__(self, lambda_mf=0.1, alpha_weights=[0.5, 0.2, 0.3]):
        super(MSLoss, self).__init__()
        self.lambda_mf = lambda_mf
        self.alpha_weights = alpha_weights
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_list, target):
        # target is the ground truth image
        # pred_list is a tuple of (E1, E2, E3) from the decoder

        # --- Content Loss (Gamma_con) ---
        # The paper says E_m are outputs from the m-th layer of the decoder.
        # We assume pred_list correspond to E1, E2, E3.
        # pred_list[0] -> output from decoder1
        # pred_list[1] -> output from decoder2
        # pred_list[2] -> output from decoder3
        
        # Resize targets to match pred_list resolutions
        target_resized1 = F.interpolate(target, size=pred_list[0].shape[-2:], mode='bilinear', align_corners=False)
        target_resized2 = F.interpolate(target, size=pred_list[1].shape[-2:], mode='bilinear', align_corners=False)
        target_resized3 = F.interpolate(target, size=pred_list[2].shape[-2:], mode='bilinear', align_corners=False)

        # Calculate L1 loss for each decoder output
        l1_loss1 = self.l1_loss(pred_list[0], target_resized1)
        l1_loss2 = self.l1_loss(pred_list[1], target_resized2)
        l1_loss3 = self.l1_loss(pred_list[2], target_resized3)

        gamma_con = (self.alpha_weights[0] * l1_loss1 +
                     self.alpha_weights[1] * l1_loss2 +
                     self.alpha_weights[2] * l1_loss3)

        # --- Auxiliary Loss (Gamma_mf) ---
        # FFT transform for output and target
        def fft_transform(image):
            # Convert to grayscale for FFT
            if image.shape[1] == 3:
                image = torch.mean(image, dim=1, keepdim=True)
            fft = torch.fft.fft2(image)
            return torch.abs(fft)

        fft_out1 = fft_transform(pred_list[0])
        fft_out2 = fft_transform(pred_list[1])
        fft_out3 = fft_transform(pred_list[2])

        fft_target1 = fft_transform(target_resized1)
        fft_target2 = fft_transform(target_resized2)
        fft_target3 = fft_transform(target_resized3)

        # L1 loss on FFT outputs
        mf_loss1 = self.l1_loss(fft_out1, fft_target1)
        mf_loss2 = self.l1_loss(fft_out2, fft_target2)
        mf_loss3 = self.l1_loss(fft_out3, fft_target3)
        
        gamma_mf = (mf_loss1 + mf_loss2 + mf_loss3) / 3

        # --- Final MSLoss ---
        ms_loss = gamma_con + self.lambda_mf * gamma_mf
        
        return ms_loss