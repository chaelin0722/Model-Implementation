
def preprocessing(codes):
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

            re.sub(r"[^a-zA-Z0-9,=+<=,-.<=~!%^()"''",==,=,!=,@]", "", cc[j])

            cc[j].strip()
            j += 1

        print(code)
        print("________________")
        # test2 = ' '.join(test2).split()
        cc = list(filter(None, cc))
        print(cc)
        #    dataset["code"][code_i] = str(cc)
        dataset['code1'][code_i].replace(dataset['code1'][code_i], str(cc))
        code_i += 1
