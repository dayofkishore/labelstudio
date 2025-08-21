### Summary
This code base customizes open source annotation tool Label-Studio.
With this, we adapt LS (label-studio) and integrate into our training/retraining pipeline. 

### Changes from previous version 
In previous versions, we only loaded the (pdf converted to) image files into LS.
In this version, we are also importing the OCR text json.
This way, we can now auto-populate the bouding boxes. 
Annotator do not manually draw the boxes. 

### Python modules
- pdf_to_images.py: Converts a given pdf into png images
- convert_to_ls.py: Converts the ocr text json into LS tasks
- convert_to_Textract: Converts the LS json back to textract raster

### LS config
- ls_config: Need to use this xml to create custom template for ls label interface.
	

### Usage 
- python pdf_to_images.py sample_data\135942130.pdf
- python convert_to_ls.py "sample_data\135942130-text.json" "sample_data\135942130-forms-model.json" "sample_data\135942130_{page}.png" --out "sample_data\135942130ls_tasks.json"

### creating project in ls
- run requirements.txt
- from cmd run "label-studio start". This will start ls loclaly at 8080
- create a project
- copy xml into custom label interface template
- import images
- copy the source for images
- update the task json in local system with these URLs
- import task json
- click into task json
- annotate, update and submit
- export json
- convert the exported json back to textract schema for training purposes



