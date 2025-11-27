import numpy as np

# ------------------- REQUIRED FUNCTIONS --------------------

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def mean_filter_manual(img, k):
    pad = k // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+k, j:j+k]
            out[i,j] = np.mean(window)

    return np.clip(out, 0, 255).astype(np.uint8)

def gradient_inverse_smoothing(img, k, p=0.2):

    if k % 2 == 0:
        raise ValueError("k must be odd")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")

    pad = k // 2
    padded = np.pad(img.astype(np.float32), pad, mode='reflect')
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)

    # Precomputing offsets in the kxk window
    offsets = []
    for dy in range(-pad, pad + 1):          
        for dx in range(-pad, pad + 1):      
            offsets.append((dy, dx))

    for i in range(H):
        for j in range(W):
            ci = i + pad
            cj = j + pad
            center_val = float(padded[ci, cj])

            # collecting u values (for neighbors only) and neighbor values
            u_vals = []
            neigh_vals = [] # (dy, dx, val)
            for (dy, dx) in offsets:
                ni = ci + dy
                nj = cj + dx
                val = float(padded[ni, nj])
                neigh_vals.append((dy, dx, val))
                if dy == 0 and dx == 0:
                    continue
                diff = abs(val - center_val)
                if diff == 0.0:
                    u = float(p)
                else:
                    u = 1.0 / diff
                u_vals.append(u)

            # computing neighbor weights from u_vals
            if len(u_vals) > 0:
                sum_u = float(np.sum(u_vals))  
            else:
                0.0 # no neighbors case
            
            h_center = float(p)  # center weight

            if sum_u <= 0.0:  # the fallback is NOT part of the algorithm — it is only a safety-net.
                # fallback: distribute (1-p) equally among neighbors
                if len(u_vals) > 0: # unexpected case
                    neighbor_weights = [(1.0 - h_center) / len(u_vals)] * len(u_vals) # equally distributed weights
                else:
                    neighbor_weights = []
            else:
                factor = (1.0 - h_center) / sum_u
                neighbor_weights = [factor * u for u in u_vals]

            # combining center and neighbor contributions
            g_val = h_center * center_val
            u_idx = 0
            for (dy, dx, val) in neigh_vals:
                if dy == 0 and dx == 0:
                    continue
                h_n = neighbor_weights[u_idx]
                g_val += h_n * val
                u_idx += 1

            out[i, j] = g_val

    return np.clip(out, 0, 255).astype(np.uint8)

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float))**2)