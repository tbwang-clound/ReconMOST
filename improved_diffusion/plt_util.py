# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os



def sample_vision(data_path, picture_path, top_k=None):
    data = np.load(data_path)
    arr = data['arr_0']
    print(np.max(arr), np.min(arr))
    os.makedirs(picture_path, exist_ok=True)  
    num_images_to_show = len(arr) if top_k is None else top_k
    for i in range(num_images_to_show):
        im = plt.imshow(arr[i].squeeze(), cmap='coolwarm', vmin=-5, vmax=40, origin='lower')  
        plt.title(f"Sample {i}")
        plt.axis('off')  

        cbar = plt.colorbar(im, orientation='vertical') 
        cbar.set_label('Temperature (°C)')  
        cbar_ticks = np.linspace(-5, 40, num=10) 
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks]) 
        
        save_path = os.path.join(picture_path, f"sample_{i}.png")  
        plt.savefig(save_path)
        plt.close()  
