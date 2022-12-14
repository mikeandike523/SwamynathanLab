# Adapted from https://medium.com/sicara/tensorflow-gpu-opencv-jupyter-docker-10705b6cd1d
# Additional suggestions by https://stackoverflow.com/a/39734201/5166365 (adding --ip=0.0.0.0)
# Usage of -v flag: https://stackoverflow.com/a/31450033/5166365



docker run -v $PSScriptRoot/src:/src -v $PSScriptRoot/assets:/assets --gpus all --name tf1 -it -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu bash -c "cd /src && ./install.sh"
if(!$?){
    Write-Host "Docker container exists or there was an error"
    docker start tf1
    if(!$?){
        Write-Host "Docker container is already started or there was an error"   
    }
    docker exec -it tf1 bash -c "cd /src&&bash"
}