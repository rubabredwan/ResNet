import torch
import matplotlib.pyplot as plt
import numpy as np

checkpoint = torch.load('bal.pth')

def plot(history):
    fig, ax = plt.subplots(2, 1, figsize = (10, 10))

    ax[0].plot(history['acc'])
    ax[0].plot(history['val_acc'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'])
    ax[0].grid(b=True, linestyle='--')
    
    ax[1].plot(history['loss'])
    ax[1].plot(history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'])
    ax[1].grid(b=True, linestyle='--')
    

    plt.savefig('plot.png', dpi=100)
    plt.show()

plot(checkpoint['history'])