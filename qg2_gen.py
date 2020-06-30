import json
import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

def show_tree(node, depth=0, with_key=""):
    if with_key != "":
        temp_str = "  "*depth+with_key+": "
    else:
        temp_str = "  "*depth
    if type(node) == type(dict()):
        print(temp_str + "dict")
        for key in node:
            show_tree(node[key], depth+1, key)
    elif type(node) == type([]):
        print(temp_str + "list")
        show_tree(node[0], depth+1)
    else:
        print(temp_str + str(type(node)))

class korquad2_viewer():
    def __init__(self, path):
        self.json = None
        self.version = None
        
        with open(path) as f:
            print(path + "- loaded")
            temp = json.load(f)
            #show_tree(temp)
            self.json = temp["data"]
            self.version = temp["version"]
        
    def __iter__(self):
        for doc in self.json:
            yield doc
"""
dict
  data: list
    dict
      raw_html: <class 'str'>
      qas: list
        dict
          question: <class 'str'>
          id: <class 'str'>
          answer: dict
            html_answer_text: <class 'str'>
            answer_start: <class 'int'>
            html_answer_start: <class 'int'>
            text: <class 'str'>
      title: <class 'str'>
      url: <class 'str'>
      context: <class 'str'>
  version: <class 'str'>
"""          
gwalho = re.compile("\[[0-9]*\]")
html_tag = re.compile("<[^>]*>")

temp_json = []

wnd = 1000
err_cnt = 0
all_cnt = 0

k2_to_k1 = {"version": "k2_aug", "data": []}
fp = open("k2k2aug.json", "w", encoding="utf8")

for data_type in [("train", 39), ("dev", 5)]:
    for i in range(0, data_type[1]):
        i = str(i)
        if len(i) == 1:
            i = "0"+i
        temp = korquad2_viewer("./"+data_type[0]+"/korquad2.1_"+data_type[0]+"_"+i+".json")

        for doc in tqdm(temp):
            for i, qas in enumerate(doc["qas"]):
                all_cnt += 1
                question = qas["question"].replace("\n", " ")
                answer = BeautifulSoup(qas["answer"]["text"], "lxml").text.replace("\n", " ")
                key_title = "%s\t%s\t%s"%(doc["title"].replace("_", " "), doc["url"], qas["id"])
                cleantext = BeautifulSoup(doc["context"], "lxml").text.replace("\n", " ")
                ##filter##########
                for tag in html_tag.findall(cleantext):
                    cleantext = cleantext.replace(tag, "")
                cleantext = cleantext.replace("\n", " ")
                cleantext = cleantext.replace("[편집]", "")
                cleantext = cleantext.replace("  ", " ").replace("  ", " ").replace("  ", " ")
                cleantext = " ".join(cleantext.split("원본 주소")[:-1])
                
                for g in gwalho.findall(cleantext):
                    cleantext = cleantext.replace(g, "")
                if answer not in cleantext:
                    err_cnt += 1
                    continue
                if len(answer) > 20:
                    err_cnt += 1
                    continue

                #answer가 들어있는 passage주변만 추출
                #sent tokenizer사용
                #title제외하고 날려버릴 부분이 많음
                    #진짜 날려야하나? sent_tokenize만으로도 많은 부분이 해결되지 않을까?
                
                cleantext = sent_tokenize(cleantext)
                answer_lst = [i for i in range(len(cleantext)) if answer in cleantext[i]]
                if len(answer_lst) > 1 or len(answer_lst) == 0 :
                    err_cnt += 1
                    continue
                
                candi_context = [cleantext[answer_lst[0]]]
                if len(cleantext[answer_lst[0]]) < 50:
                    wnd = 1 
                    candi_context = cleantext[answer_lst[0]-wnd:answer_lst[0]+wnd+1]
                    while(len(" ".join(candi_context))<50 and answer_lst[0]-wnd >= 0 and answer_lst[0]+wnd < len(cleantext)):
                        wnd += 1
                        candi_context = cleantext[answer_lst[0]-wnd:answer_lst[0]+wnd+1]
                
                    
               
                context = " ".join(candi_context)

                if len(context) > 300:
                    err_cnt += 1
                    continue
                
                title = key_title.split("\t")[0]
                if title not in context:
                    context = title + " " + context
                paragraphs = [{"qas":[{"answers":[{"text":answer, "answer_start":context.find(answer)}], "id":str(all_cnt), "question":question}], "context":context}]
                
                data = {"paragraphs":paragraphs, "title":title}
                k2_to_k1["data"].append(data)

                #print(paragraphs)
                

json.dump(k2_to_k1, fp)               
                
print(err_cnt, all_cnt)
            