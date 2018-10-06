# coding: utf-8
# TODO: REMOVE THIS FILE
import glimpse
import torch
import scipy.misc
import matplotlib.pyplot as plt

x = scipy.misc.imread('/home/zy1404/a.jpg') #('/home/gq/Pictures/a.jpg')
x = x / 255.
x = x.transpose(2, 0, 1)
x = torch.FloatTensor(x)
x = x[None]
plt.imshow(x[0].numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()

glm = glimpse.GaussianGlimpse((3, 3))

a_rel = torch.FloatTensor([[0.5, 0.5, 1, 1, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.25, 0.25]]).requires_grad_()
a = glm._to_absolute_attention(a_rel[None], x.shape[-2:])

cx, cy, dx, dy, sx, sy = torch.unbind(a, -1)
batch_size, n_glims, _ = a.size()
_, nchannels, nrows, ncols = x.size()
n_glim_rows, n_glim_cols = 3, 3
Fy = glimpse.gaussian_masks(cy, dy, sy, nrows, n_glim_rows)
Fy_inv = glimpse.inverse_gaussian_masks_surrogate(cy, dy, sy, n_glim_rows, nrows)
#Fy_inv = glimpse.upsampling_masks(cy, dy, sy, n_glim_rows, nrows)

Fx = glimpse.gaussian_masks(cx, dx, sx, ncols, n_glim_cols)
Fx_inv = glimpse.inverse_gaussian_masks_surrogate(cx, dx, sx, n_glim_cols, ncols)
#Fx_inv = glimpse.upsampling_masks(cx, dx, sx, n_glim_cols, ncols)

plt.imshow(Fy[0, 1].detach())
plt.show()
plt.imshow(Fy_inv[0, 1].detach())
plt.show()

Fy = Fy.unsqueeze(2)
Fx = Fx.unsqueeze(2)
Fy_inv = Fy_inv.unsqueeze(2)
Fx_inv = Fx_inv.unsqueeze(2)


x = x.unsqueeze(1)
g = Fy.transpose(-1, -2) @ x @ Fx
plt.imshow(g[0, 0].detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()
plt.imshow(g[0, 1].detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()

# concat an alpha channel
g = torch.cat([g, torch.ones(batch_size, n_glims, 1, n_glim_rows, n_glim_cols)], 2)

x_inv = Fy_inv.transpose(-1, -2) @ g @ Fx_inv

# To avoid weirdness in viz I clipped the values to between 0 and 1.
# No need to do that in feature maps I guess.
plt.imshow(x_inv.clamp(min=0, max=1)[0, 0].detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()
plt.imshow(x_inv.clamp(min=0, max=1)[0, 1].detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()

# assuming that x_inv[:, 0] is the old image and x_inv[:, 1] is the new image
# we wish to overlay atop the old one
x_old, x_old_alpha = x_inv[:, 0, :-1], x_inv[:, 0, -1]
x_new, x_new_alpha = x_inv[:, 1, :-1], x_inv[:, 1, -1]

print(x_new_alpha)
plt.imshow(x_new_alpha[0].detach(), cmap='gray')

# We assume that the old image is *always* opaque (i.e. alpha = 1)
# https://en.wikipedia.org/wiki/Alpha_compositing
x_inv_overlay = x_new * x_new_alpha + x_old * (1 - x_new_alpha)
plt.imshow(x_inv_overlay.clamp(min=0, max=1)[0].detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()
