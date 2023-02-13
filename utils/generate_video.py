import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.animation as animation


def generate_video(img_folder_path: str, video_name: str = 'video')-> None:
    """
    Args:
        - img_folder_path: e.g. 'images/02-06-12-46'
        - video_name: e.g. 'video'
    """

    files = [f for f in listdir(img_folder_path) if isfile(join(img_folder_path, f))]
    files.sort()
    # files = files[:300]

    imgs = []

    for file in files:
        img = mpimg.imread(os.path.join(img_folder_path, file))
        imgs.append(img)

    frames = []  # for storing the generated images
    fig = plt.figure()
    plt.axis('off')
    for img in imgs:
        frames.append([plt.imshow(img, cmap=cm.Greys_r,animated=True)])

    del imgs

    ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=1000)
    video_name = f'{video_name}.mp4'
    os.makedirs('videos', exist_ok=True)
    ani.save(os.path.join('videos', video_name))
    
    return None


if __name__ == '__main__':
    now = "03-06-20-24"
    generate_video(os.path.join('images', now), video_name=now)