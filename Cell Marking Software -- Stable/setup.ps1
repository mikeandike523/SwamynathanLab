try{
    conda env create -f cellmarkingsoftware.yml
}catch{
   Write-Host "Conda env may already exist."
   try{
    conda env update -f cellmarkingsoftware.yml
   }
   catch{
    Write-Host "Could not update conda env. Recreating..."
    conda env remove -n cellmarkingsoftware
    conda env create -f cellmarkingsoftware.yml
   }
}
python setup.py
pause