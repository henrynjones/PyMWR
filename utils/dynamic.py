import torch

def dynamic(M):
    # Find the minimum and maximum values in the tensor
    minV = torch.min(M)
    maxV = torch.max(M)

    # Return the result as a 1D tensor
    out = torch.tensor([minV.item(), maxV.item()], dtype=M.dtype, device=M.device)
    return out

# Example usage:
# M = torch.randn((10, 10))  # Example input
# result = dynamic(M)
# print(result)