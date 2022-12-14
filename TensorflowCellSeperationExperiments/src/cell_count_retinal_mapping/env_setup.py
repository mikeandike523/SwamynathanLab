def env_setup():
    print("cell_count_retinal_mapping.env_setup")
    import sys
    import os
    sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
    # os.chdir(os.path.dirname(os.path.dirname(__file__)))