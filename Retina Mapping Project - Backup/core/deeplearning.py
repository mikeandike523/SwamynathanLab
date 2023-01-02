import cv2
import numpy as np

import utils as U
import image_processing as IP
import u_net as UNET

class CellDifferentiator:

    # Dropout layer is not included in inference, so it is okay if the training model has a dropout layer and the prediction model does not
    def __init__(self,size,nBaseFilters,weights_filepath):
        self.size=size
        self.nBaseFilters=nBaseFilters
        self.weights_filepath=weights_filepath
        self.model = UNET.UNet(size,nBaseFilters,None)
        self.model.load_weights(weights_filepath)

    def apply_to_images(self,list_of_input_images):

        # Apply the neural network to a batch of images
        # The type of list_of_input_images is a list of RGB numpy arrays, or an array of shape (-1,H,W,3)

        input_batch = []

        nontrivial_idxs = []

        for idx, input_image in enumerate(list_of_input_images):

            if not (np.all(input_image == np.mean(input_image))):
            
                nontrivial_idxs.append(idx)
                resized_image = cv2.resize(input_image.copy(),dsize=(self.size,self.size),interpolation=cv2.INTER_CUBIC)
                input_batch.append(resized_image.astype(float)/255)

        if len(input_batch) == 0:
            return [np.zeros_like(list_of_input_images[i]) for i in range(len(list_of_input_images))]

        input_batch = np.array(input_batch,float)

        output_batch = self.model.predict_on_batch(input_batch)

        #print(output_batch.shape)
        #(1, 128, 128, 3, 1)
        # Where does the extra 1 come from?
        # What happens when I input 2 images?
        #(2, 128, 128, 3, 1)
        # So it seems as though the last axis has to be squeezed
        # print(output_batch.shape)

        output_batch = output_batch.squeeze(axis=-1)

        list_of_output_images = []

        output_batch_pointer = 0

        for idx in range(len(list_of_input_images)):
      
            if idx in nontrivial_idxs:

                output_image = output_batch[output_batch_pointer]
                output_batch_pointer +=1

                H, W = list_of_input_images[idx].shape[:2]

                list_of_output_images.append(cv2.resize((255*output_image).astype(np.uint8),dsize=(W,H),interpolation=cv2.INTER_CUBIC))

            else:

                list_of_output_images.append(np.zeros_like(list_of_input_images[idx]))

        return list_of_output_images       