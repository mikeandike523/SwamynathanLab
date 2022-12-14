import tempfile

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import PIL.Image

import utils as U

def export_mp4_from_images(images,framerate,output_path):

    assert output_path.endswith('.mp4')

    tempdir = tempfile.TemporaryDirectory()
    # U.init_folder(tempdir.name,clear=True) # Should not be needed
    image_filenames = []

    for idx, image in enumerate(images):

        PIL.Image.fromarray(image).save(f"{tempdir.name}/frame{idx}.png")
        image_filenames.append(f"{tempdir.name}/frame{idx}.png")

    movie_clip = ImageSequenceClip(image_filenames,framerate)

    movie_clip.write_videofile(output_path)

    
    