import os

def modify_filename(path,filename):
	if os.path.exists(path+filename):
		str_list = filename.split(".")
		file = str_list[0]
		file_list = file.split("_")
		if file_list[-1].isdigit():
			fileint = int(file_list[-1])
			fileint += 1
			file_list[-1] = "{0}".format(fileint)
		else:
			fileint = 1
			file_list.append("{0}".format(fileint))
		new_file = "_".join(file_list)
		str_list[0] = new_file
		new_filename = ".".join(str_list)
	else:
		new_filename = filename
	return path+new_filename