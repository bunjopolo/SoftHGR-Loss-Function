import torch

def trace3d(matrix):
    """
    Calculates the 3D trace of a matrix.

    Args:
    - matrix (torch.Tensor): Input 3D matrix.

    Returns:
    - torch.Tensor: 3D trace of the matrix.
    """
    mask = torch.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[2])).to(matrix.device)
    mask[:, torch.arange(0, 3), torch.arange(0, 3)] = 1.0

    output = matrix * mask
    output = torch.sum(output, axis=(1, 2)).sum()
    return output

def SoftHgr3D(x, y):
    """
    Computes the Soft  Hirschfeld-Gebelein-Renyi (HGR) maximal correlation (SoftHHR) objective function for # channel RGB data.

    Args:
    - x (torch.Tensor): Input data tensor.
    - y (torch.Tensor): Input data tensor.

    Returns:
    - torch.Tensor: SoftHgr objective function value.
    """
    m = x.size(0)

    # mean center the data
    f = x - torch.mean(x)
    g = y - torch.mean(y)

    cov_f = torch.zeros(f.size(1), f.size(2), f.size(3)).to(f.device)
    cov_g = torch.zeros(g.size(1), g.size(2), g.size(3)).to(f.device)
    hgr_objective = 0

    # compute covariance matrices
    for f_xi, g_yi in zip(f, g):
        transpose_f = f_xi.permute(0, 2, 1)
        transpose_g = g_yi.permute(0, 2, 1)
        cov_f += torch.matmul(f_xi, transpose_f) / (m - 1)
        cov_g += torch.matmul(g_yi, transpose_g) / (m - 1)

    # compute hgr objective function
    for f_xi, g_yi in zip(f, g):
        transpose_f = f_xi.permute(0, 2, 1)
        # added max
        hgr_objective += torch.max(torch.matmul(transpose_f, g_yi)) - 0.5 * trace3d(torch.matmul(cov_f, cov_g))
    hgr_objective = hgr_objective / (m - 1)

    return hgr_objective
