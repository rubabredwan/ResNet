import torch
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(15, 15))

for net in ('ResNet20', 'ResNet32', 'ResNet44', 'ResNet56'):
    checkpoint = torch.load('pretrained/' + net + '.pth')
    history = checkpoint['history']
    c = ax[0].plot(history['acc'], linewidth=0.5, label=net)
    ax[0].plot(history['val_acc'], linewidth=0.5, color=c[0].get_color())
    c = ax[1].plot(history['loss'], linewidth=0.5, label=net)
    ax[1].plot(history['val_loss'], linewidth=0.5, color=c[0].get_color())

ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend()
ax[0].grid(b=True, linestyle='--')

ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend()
ax[1].grid(b=True, linestyle='--')
    
plt.savefig('plot.png', dpi=100)
