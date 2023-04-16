import os

def make_data(files_path, output_file_path):
    """This function will get all the files in the files_path directory and add them up into one file"""
    
    all_files = os.listdir(files_path)
    complete_dickens = open(output_file_path, "w")
    
    for f in all_files:
        all_lines = open("/datasets/charles_dickens/" + f, "r").read()
        
        starting_tag = "Title:"
        if starting_tag in all_lines:
            all_lines = all_lines.split(starting_tag)[1]
        
        end_tag = "END OF THE PROJECT GUTENBERG EBOOK"
        if end_tag in all_lines:
            all_lines = all_lines.split(end_tag)[0]
            complete_dickens.write(all_lines)
    
    print("Data Formed Correctly!")
    complete_dickens.close()
    
make_data("/dickens/", "data_files/dickens_in_one_place.txt")