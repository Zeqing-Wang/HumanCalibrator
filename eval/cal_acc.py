import json
import re

def find_word_in_string(word_list, input_string):
    for word in word_list:
        if word in input_string:
            return word  
    return None 

if __name__ == '__main__':
    res_json = './baseline/llava-v1.6-34b/coco_val.json'
    res = json.load(open(res_json, "r"))
    
    total = 0
    acc_count = 0
    text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear']
    for key in res:
        for key_image in res[key]:
            total += 1
            label = find_word_in_string(word_list=text_list, input_string=key_image)
            # print(res[key][key_image])
            question_key = list(res[key][key_image]['question_ans'].keys())[0]
            pred = find_word_in_string(word_list=text_list, input_string=res[key][key_image]['question_ans'][question_key].lower())
            if pred != label:
                continue
            else:
                acc_count += 1
    print(f'{res_json} acc:\n', acc_count/total)
    pass
