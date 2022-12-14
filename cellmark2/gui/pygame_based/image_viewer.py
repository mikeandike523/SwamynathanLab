import os

import pygame
import numpy as np

from core.io import imload
from core.image_ops import scale
from core.python_utils import fluiddict

MAX_ZOOM_LEVEL = 15
MIN_ZOOM_LEVEL = 0.125
ZOOM_RATE = 0.125

class ImageViewer:
    
    def __init__(self, pixels=None):
        
        if pixels is None:
            pixels = scale(imload(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'logo.webp'
            )),5.0)
        
        self.pixels = pixels
        self.h, self.w, self.nchannels =  self.pixels.shape

        self.ui_state = fluiddict(lambda key: None)

    def on_scroll(self, delta):

        delta = python_utils.signum(delta)

        current_zoom_level = self.ui_state.unwrap_or_value("zoom_leve", 1.0)

        new_zoom_level = python_utils.clamp(current_zoom_level-delta*ZOOM_RATE,MIN_ZOOM_LEVEL, MAX_ZOOM_LEVEL)

        self.ui_state.zoom_level = new_zoom_level

    def ui_loop(self):

        lmb_down, mmb_down, rmb_down = pygame.mouse.get_pressed()

        mouse_x, mouse_y = pygame.mouse.get_pos()

        dragging = self.ui_state.unwrap_or_value("dragging",False)

        if rmb_down:
            if dragging:
                init_mouse_x, init_mouse_y = \
                    self.ui_state.unwrap_or_value(
                        "drag_init_mouse_pos",
                        np.array((0,0),int)
                    )
                delta_x = mouse_x - init_mouse_x
                delta_y = mouse_y - init_mouse_y

                zoom_level = self.ui_state.unwrap_or_value(
                    "zoom_level",
                    1.0
                )

                scaled_delta_x = delta_x / zoom_level
                scaled_delta_y = delta_y/ zoom_level

                init_origin = self.ui_state.unwrap_or_value(
                    "drag_init_origin",
                    np.array((self.w/2,self.h/2),int)
                )

                self.ui_state.origin = \
                    init_origin+np.array((
                        scaled_delta_x,
                        scaled_delta_y
                    ),int)

            else:
                self.ui_state.dragging=True
                self.ui_state.drag_init_mouse_pos = np.array((mouse_x,mouse_y),int)
                self.ui_state.drag_init_origin = self.ui_state.unwrap_or_value("origin",np.array((self.w/2,self.h/2),int))
        else:
            self.ui_state.dragging = False

    def map_screen_pixel_to_image_pixel(self, screen_x, screen_y):

        zoom_level = self.ui_state.unwrap_or_value("zoom_level", 1.0)
        origin = self.ui_state.unwrap_or_value("origin", np.array((self.w/2,self.h/2),int))

        screen_offset_x = screen_x - self.w//2
        screen_offset_y = screen_y - self.h//2

        screen_offset_x_scaled = screen_offset_x / zoom_level

        screen_offset_y_scaled = screen_offset_y / zoom_level

        image_x = int(origin[0]+screen_offset_x_scaled)
        image_y = int(origin[1]+screen_offset_y_scaled)

        return image_x, image_y

    def map_image_pixel_to_screen_pixel(self, image_x, image_y):
        pass

    def get_view_surface(self):
        
        display_pixels = np.zeros((self.h, self.w, self.nchannels),np.uint8)

        for x in range(self.w):
            for y in range(self.h):
                image_x, image_y = self.map_screen_pixel_to_image_pixel(x,y)
                pixel = self.pixels[image_y,image_x,:]
                display_pixels[y, x] = pixel

        return pygame.surfarray.make_surface(display_pixels.transpose((1,0,2)))
    
    def start(self):
        
        self.running=True
        
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.w, self.h))
        
        while self.running:

            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEWHEEL:
                    self.on_scroll(event.y)
                    
            self.ui_loop()

            self.screen.blit(self.get_view_surface(), (0, 0))
                    
            pygame.display.update()
                
        pygame.quit()