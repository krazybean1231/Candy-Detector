from PIL import Image
import glob
import os
import pillow_heif
from PIL import ImageFile
pillow_heif.register_heif_opener()
folder = r"/Users/visitor/Documents/Hackgwinnett26/data-preliminary/Kiss/"
#outputdir = r"/Users/visitor/Documents/Hackgwinnett26/data/Snickers/"
i=0
ImageFile.LOAD_TRUNCATED_IMAGES = True
with os.scandir(folder) as entries:
  for entry in entries:
    
    if entry.is_file():
        try:
            file_name, _ = os.path.splitext(entry)
            if file_name == ".DS_Store":
                continue
            img = Image.open(entry)
        
            # Save with the new .png extension
            img.save(f"{i}.png", "PNG")
            print(f"Converted {entry} to {file_name}.png")
            i+=1


        except (Image.UnidentifiedImageError, OSError):
            continue

        