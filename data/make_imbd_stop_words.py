with open("train.txt", 'r') as content_file:
        nums = set()
        lines = content_file.readlines()
        with open("stop_words_imbd/stop_words_with_nums.txt", 'a') as text:
            for line in lines:
                l = line.split(",")
                for word in l[0].split(" "):
                    if word.isdigit():
                        #print(word)
                        if not (word in nums):
                            text.write(word+",")
                            print(word)
                            nums.add(word)
                        