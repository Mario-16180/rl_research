import os
import imageio

def save_frames_as_gif(frames, path='experimenting/saved_renders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')