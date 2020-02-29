import os
for word in ["test","train"]:
    f = open("/Users/kaan/Downloads/aclImdb/"+word+".txt", "w")
    for op in ["pos","neg"]:
        path = "/Users/kaan/Downloads/aclImdb/"+word+"/"+op
        for filename in os.listdir(path):
            temp = filename.split("_")
            score = ""
            for c in temp[1]:
                if c==".":break
                else: score+=c
            with open(path+'/'+filename, 'r') as content_file:
                content = content_file.read()
                f.write(content.replace("<br /><br />","").replace(",","") +","+score+ os.linesep)
                
    f.close()
            
            
            
