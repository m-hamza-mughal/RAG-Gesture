import json
import numpy as np

def parse_discourse_tokens(ann_json_path):
    # breakpoint()
    with open(ann_json_path, "r") as f:
        ann = json.load(f)

    all_tokens = []
    for sent in ann["sentences"]:
        all_tokens += sent["tokens"]
        
    # tokens = ann["sentences"][0]["tokens"]
    text = []
    start = []
    end = []
    duration = []
    for token in all_tokens:
        token["surface"] = token["surface"].replace(' ', '')
        text.append(token["surface"])
        start.append(token["startSec"])
        end.append(token["endSec"])
        duration.append(token["endSec"] - token["startSec"])

    return {
        "text": np.array(text),
        "start": np.array(start),
        "end": np.array(end),
        "duration": np.array(duration)
    }

def parse_discourse_relations(file_json, start, end):
    ann = file_json
    
    relations = ann["relations"]
    all_tokens = []
    for sent in ann["sentences"]:
        all_tokens += sent["tokens"]

    for token in all_tokens:
        token["surface"] = token["surface"].replace(' ', '')
    
    connectives = []
    for relation in relations:
        conn = {}
        # breakpoint()
        conn["connective"] = relation["Connective"]["RawText"]
        min_token = min(relation["Connective"]["TokenList"] + relation["Arg1"]["TokenList"] + relation["Arg2"]["TokenList"])
        max_token = max(relation["Connective"]["TokenList"] + relation["Arg1"]["TokenList"] + relation["Arg2"]["TokenList"])

        conn_start_tk = min(relation["Connective"]["TokenList"])
        # conn_start_sec = ann["sentences"][0]["tokens"][conn_start_tk]["startSec"]
        conn_start_sec = all_tokens[conn_start_tk]["startSec"]
        conn_end_tk = max(relation["Connective"]["TokenList"])
        # conn_end_sec = ann["sentences"][0]["tokens"][conn_end_tk]["endSec"]
        conn_end_sec = all_tokens[conn_end_tk]["endSec"]

        # we need the connective to be inside the start and end
        # if conn_end_sec < start or conn_start_sec > end:
        #     continue       
        if not (conn_start_sec >= start and conn_end_sec <= end):
            continue 

        # relation start and end
        # conn["start"] = max(ann["sentences"][0]["tokens"][min_token]["startSec"], start)
        conn["start"] = max(all_tokens[min_token]["startSec"], start)
        # conn["end"] = min(ann["sentences"][0]["tokens"][max_token]["endSec"], end)
        conn["end"] = min(all_tokens[max_token]["endSec"], end)

        # connective start and end
        conn["conn_start"] = max(conn_start_sec, start)
        conn["conn_end"] = min(conn_end_sec, end)

        conn["sense"] = relation["Sense"][0]

        # TODO: make sure arg1 text is inside start and end
        conn['Arg1'] = {}
        # conn['Arg1']['text'] = relation["Arg1"]["RawText"]
        if len(relation["Arg1"]["TokenList"]) == 0:
            conn['Arg1']['start'] = conn["start"]
            conn['Arg1']['end'] = conn["start"]
            conn['Arg1']['text'] = ""
        else:
            # conn['Arg1']['start'] = ann["sentences"][0]["tokens"][relation["Arg1"]["TokenList"][0]]["startSec"]
            conn['Arg1']['start'] = all_tokens[relation["Arg1"]["TokenList"][0]]["startSec"]
            conn['Arg1']['start'] = max(conn['Arg1']['start'], start)
            # conn['Arg1']['end'] = ann["sentences"][0]["tokens"][relation["Arg1"]["TokenList"][-1]]["endSec"]
            conn['Arg1']['end'] = all_tokens[relation["Arg1"]["TokenList"][-1]]["endSec"]
            conn['Arg1']['end'] = min(conn['Arg1']['end'], end)

            # get the text which is inside the start and end
            arg1_text = []
            for t_idx in relation["Arg1"]["TokenList"]:
                # token = ann["sentences"][0]["tokens"][t_idx]
                token = all_tokens[t_idx]
                if token["startSec"] >= conn['Arg1']['start'] and token["endSec"] <= conn['Arg1']['end']:
                    arg1_text.append(token["surface"])
            conn['Arg1']['text'] = " ".join(arg1_text)
            conn['Arg1']['text'] = conn['Arg1']['text'] #.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').replace(' :', ':').replace(' ;', ';').replace(' -', '-').replace(' \'', '\'').replace('  ', ' ')


        conn['Arg2'] = {}
        # conn['Arg2']['text'] = relation["Arg2"]["RawText"]
        if len(relation["Arg2"]["TokenList"]) == 0:
            conn['Arg2']['start'] = max(conn["end"], conn['Arg1']['end'])
            conn['Arg2']['end'] = max(conn["end"], conn['Arg1']['end'])
            conn['Arg2']['text'] = ""
        else:
            # conn['Arg2']['start'] = ann["sentences"][0]["tokens"][relation["Arg2"]["TokenList"][0]]["startSec"]
            conn['Arg2']['start'] = all_tokens[relation["Arg2"]["TokenList"][0]]["startSec"]
            conn['Arg2']['start'] = max(conn['Arg2']['start'], start)
            # conn['Arg2']['end'] = ann["sentences"][0]["tokens"][relation["Arg2"]["TokenList"][-1]]["endSec"]
            conn['Arg2']['end'] = all_tokens[relation["Arg2"]["TokenList"][-1]]["endSec"]
            conn['Arg2']['end'] = min(conn['Arg2']['end'], end)

            # get the text which is inside the start and end
            arg2_text = []
            for t_idx in relation["Arg2"]["TokenList"]:
                # token = ann["sentences"][0]["tokens"][t_idx]
                token = all_tokens[t_idx]
                if token["startSec"] >= conn['Arg2']['start'] and token["endSec"] <= conn['Arg2']['end']:
                    arg2_text.append(token["surface"])
            conn['Arg2']['text'] = " ".join(arg2_text)
            conn['Arg2']['text'] = conn['Arg2']['text'] #.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').replace(' :', ':').replace(' ;', ';').replace(' -', '-').replace(' \'', '\'').replace('  ', ' ')

        connectives.append(conn)


    return connectives