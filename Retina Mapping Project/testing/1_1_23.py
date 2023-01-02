from core.setup import setup
setup()

import utils as U
import image_processing as IP

print(U.Geometry.get_window_locations_covering_image(5,5,2,2,0.0,0.0))