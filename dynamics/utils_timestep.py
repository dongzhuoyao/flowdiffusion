import torch




def sample_r_t_logit_normal(x1, mu, sigma):
    def _sample_t(x1, mu, sigma):
        x = torch.randn(len(x1), device=x1.device) * sigma + mu
        t = torch.sigmoid(x)
        return t
    t = _sample_t(x1, mu, sigma)
    r = _sample_t(x1, mu, sigma)
    # Stack and sort
    combined = torch.stack([t, r], dim=0)  # Shape: [2, batch_size]
    sorted_vals, _ = torch.sort(combined, dim=0)  # Sort along first dimension
    
    # Unpack the sorted values
    t, r = sorted_vals[0], sorted_vals[1]  # t will be smaller than r
    return t,r

def cosmap_sample(num_samples):
    """
    Sample timesteps t using CosMap schedule.

    Args:
        num_samples (int): Number of samples to generate.

    Returns:
        torch.Tensor: Sampled timesteps t in (0, 1)
    """
    u = torch.rand(num_samples)
    t = 1 - 1 / (torch.tan(torch.pi / 2 * u) + 1)
    return t



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t_cosmap_sample = cosmap_sample(100_00)
    plt.hist(t_cosmap_sample, bins=100)
    plt.savefig("cosmap_sample.png")
