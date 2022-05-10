
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
import re
import re

#text = "abc123def456ghi中國語<=+,"
#new_text = re.sub(r"[^a-zA-Z0-9,=+<=,-.<=~!%^()"''",==,=,!=,]", "", text)
#print(new_text)
test =  "most_common(2)で最大の2個とりだせるお a[0][0]"
MODEL = "klue/roberta-base"
MAX_LEN = 256

## def

code = test

code = re.sub(r"[^a-zA-Z0-9,=+<=,-.<=~!%^()"''",==,=,!=,@]", "", code)

print(code)

def preprocessing(code):
    cc = code.split("\n")

    text_list = [k for k, value in enumerate(cc) if value == '"""']  # [1 ,3]

    if len(text_list) != 0 and len(text_list) > 1:
        for i in range(text_list[0], text_list[1] + 1):
            cc[i] = ""
    i = 0

    j = 0
    for t in cc:
        if t.startswith('#'):
            cc[j] = ""

        if '"""' in t:
            cc[j] = ""

        cc[j] =  re.sub(r"[^a-zA-Z0-9,=+<=,-.<=~!%^()"''",&&,<<,>>,<.>,==,=,!=,||,@*]", "", "", cc[j])

        cc[j] = cc[j].strip()
        j += 1

    print(code)
    print("________________")
    # test2 = ' '.join(test2).split()
    cc = list(filter(None, cc))
    out = ''.join(cc)
    print(out)
    return out
        #    dataset["code"][code_i] = str(cc)


def preprocessing_df(codes):
    code_i = 0
    for code in codes:
        cc = code.split("\n")

        text_list = [k for k, value in enumerate(cc) if value == '"""']  # [1 ,3]

        if len(text_list) != 0 and len(text_list) > 1:
            for i in range(text_list[0], text_list[1] + 1):
                cc[i] = ""
        i = 0

        j = 0
        for t in cc:
            if t.startswith('#'):
                cc[j] = ""

            if '"""' in t:
                cc[j] = ""

            cc[j] = re.sub(r"[^a-zA-Z0-9,=+<=,-.<=~!%^()"''",&&,<<,>>,<.>,==,=,!=,||,@*]", "", "", cc[j])

            cc[j].strip()
            j += 1

        print(code)
        print("________________")
        # test2 = ' '.join(test2).split()
        cc = list(filter(None, cc))  # erase blank line
        print(cc)
        #    dataset["code"][code_i] = str(cc)
        code_i += 1


dataset = load_dataset("csv", data_files=INPUT)['train']

preprocessing_df(dataset['code1'])


