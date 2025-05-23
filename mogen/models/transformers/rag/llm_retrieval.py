import torch
import numpy as np
# from huggingface_hub import InferenceClient
import json
import glob
import re
import copy
import time
import random

from .utils import sort_sidx_by_textsimilarity, get_word_similarity_score, map_conns_to_prominence


from dotenv import load_dotenv


from openai import OpenAI
import os

hf_infclient = None
# hf_infclient = InferenceClient(
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     token="YOUR_TOKEN_HERE",
# )

GEST_TYPE_EXP = """
1. **Beat Gesture**: A beat gesture is a brief, light touch or tap on a surface, usually accompanied by a slight movement of the hand or finger. It serves as a way to punctuate or emphasize a gesture, drawing attention to it. Beat gestures can signify:
	* Emphasis or accentuation of a statement
	* Encouragement or approval
	* Synchronization or coordination with others
2. **Metaphoric Gesture**: A metaphoric gesture uses a physical action to represent a concept or idea, often creating a vivid mental image. It can be used to explain, illustrate, or extend a metaphor. Metaphoric gestures signify:
	* The connection between the physical and conceptual realms
	* The extension of an idea or concept beyond its literal meaning
	* The creation of a vivid, memorable mental image
3. **Iconic Gesture**: An iconic gesture mimics the action or shape of the thing being described or referred to. It's a way to visualize or illustrate an object, action, or idea through a corresponding physical movement. Iconic gestures signify:
	* The connection between the physical and conceptual realms
	* The ability to convey complex information through a simple action
	* The emphasis on the similarity between the gesture and the referenced concept
4. **Deictic Gesture**: A deictic gesture points or directs attention to a specific location, object, or person. It serves as a way to indicate or specify a target, often in a shared communicative context. Deictic gestures signify:
	* Direction or indication
	* Identification or specification of a target
	* Coordination with others, referencing a shared context
"""

GEST_TYPE_EXP_SHORT = """
You are an expert in human gestures. You need to identify words that may elicit semantically meaningful gestures(deictic, iconic, metaphoric) and their types:

Metaphoric Gesture: Represents abstract ideas or concepts physically, creating a vivid mental image.
Iconic Gesture: Mimics the shape or action of the object or concept being described.
Deictic Gesture: Points to or indicates a person, object, or location.

Format your response as a python list of python tuples of (word, type). For example: [('hello', 'beat'), ('world',
'iconic')]
"""
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_llm_output(text, model="gpt-4o-mini"):
    if model == "gpt-4o-mini":
        return call_gpt_4o_mini(text)
    elif model == "llama3-8b":
        return call_llama3_8b(text)



def call_gpt_4o_mini(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": GEST_TYPE_EXP_SHORT
            },
            {
                "role": "user",
                "content": f"identify at most 2 important words which are more likely to elicit semantically meaningful gestures and what are types of those gestures in following text: \"{text}\"."
            }
        ]
    )

    return completion.choices[0].message.content


def call_llama3_8b(text):
    """
    Get the output of the LLM model
    """
    # get the output of the LLM model
    llm_text_output = ""

    for message in hf_infclient.chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"You are an expert on human gestures and know which words elicit more semantically meaningful gestures and what type of gestures they elicit. Type of gestures include (beat, metaphoric, iconic, deictic). Also here is an explanation of each type: {GEST_TYPE_EXP}",
            },
            {
                "role": "user",
                "content": f"identify which words are more likely to elicit semantically meaningful gestures and what are types of those gestures in following text: \"{text}\" \n Format your response as a python list of python tuples of (word, type). For example: [('hello', 'beat'), ('world', 'iconic')]",
            },
        ],
        max_tokens=500,
        stream=True,
    ):
        
        msg = message.choices[0].delta.content
        llm_text_output += msg if msg is not None else ""
        if msg is None:
            print("LLM output is None, waiting for 60 seconds...")
            time.sleep(60)

    return llm_text_output


def parse_gesture_labels_from_llm_output(llm_output):
    """
    Parse the gesture labels from the LLM output
    """
    gesture_labels = []
    # llm_output = llm_output.split("\n")
    match_regex = r"[\"\']*([\w \-\']+\w)[\"\']*\,\s*[\"\']*(?P<gesttype>b*eat|m*etaphoric|iconic|deictic)"
    matches = re.finditer(match_regex, llm_output, re.MULTILINE)
    for match in matches:
        if "etaphoric" in match.group("gesttype"):
            gesttype = "metaphoric"
        elif "eat" in match.group("gesttype"):
            gesttype = "beat"
        elif "iconic" in match.group("gesttype"):
            gesttype = "iconic"
        elif "deictic" in match.group("gesttype"):
            gesttype = "deictic"
        else:
            raise ValueError(f"Unknown gesture type: {match.group('gesttype')}")

        gesture_labels.append({"word": match.group(1).strip(), "name": gesttype})

    # remove the beat gesture labels (only keep the semantically meaningful ones)
    gesture_labels = [gl for gl in gesture_labels if gl["name"] != "beat"]

    
    # remove duplicates from the gesture labels because sometimes
    # regex picks up pairs from explanation section
    unique_gesture_labels = []
    for gl in gesture_labels:
        if gl in unique_gesture_labels:
            continue
        unique_gesture_labels.append(gl)

    return unique_gesture_labels


def llm_retrieval(
    text,
    text_times,
    speaker_id,
    prominence,
    db_idx_2_gesture_labels,
    db_idx_2_prominence,
    encoded_text,
    text_feat_cache,
):
    """
    retrieval function for LLM generated gesture labels
    """
    d_bounds = {}
    sample_indexes = {}
    query_gest_bounds = {}

    if text.strip() == "":
        return sample_indexes, d_bounds, query_gest_bounds

    # extract the gesture labels from the LLM model
    llm_output = get_llm_output(text)
    gesture_labels = parse_gesture_labels_from_llm_output(llm_output)

    if len(gesture_labels) == 0:
        return sample_indexes, d_bounds, query_gest_bounds

    query_gest_types = [g["name"] for g in gesture_labels]
    query_gest_words = [g["word"].lower() for g in gesture_labels]
    query_gest_words = [
        "".join([c for c in w if c.isalnum() or c.isspace()]) for w in query_gest_words
    ]

    query_gest_bounds = {}
    residual_qwords = copy.deepcopy(query_gest_words)
    for t_idx, t_time in enumerate(text_times):

        t_word = t_time[1].lower()
        t_word = "".join([c for c in t_word if c.isalnum() or c.isspace()])
        t_start, t_end = t_time[0][0], t_time[0][1]

        for q_w_idx, q_word in enumerate(query_gest_words):
            if residual_qwords[q_w_idx] is None:
                continue

            q_word = q_word.lower()
            # q_word = "".join([c for c in q_word if c.isalnum() or c.isspace()])
            if q_word == t_word or t_word in q_word.split():
                if q_w_idx not in query_gest_bounds:
                    query_gest_bounds[q_w_idx] = []

                q_type = query_gest_types[q_w_idx]
                query_gest_bounds[q_w_idx].append((q_word, q_type, t_start, t_end))
                if q_word == t_word or t_word == q_word.split()[-1]:
                    residual_qwords[q_w_idx] = None
                break

    
    if len(query_gest_bounds) == 0:
        print("No gesture bounds found\n", text, "\n", text_times)
        return sample_indexes, d_bounds, query_gest_bounds
    

    # merge the gesture boundaries
    for q_gest_word_idx, q_gest_bounds in query_gest_bounds.items():
        if len(q_gest_bounds) > 1:
            start = min([b[2] for b in q_gest_bounds])
            end = max([b[3] for b in q_gest_bounds])
            q_word = q_gest_bounds[0][0]
            q_type = q_gest_bounds[0][1]

            assert (
                q_word == query_gest_words[q_gest_word_idx]
                or q_word in query_gest_words[q_gest_word_idx].split()
            )
            assert q_type == query_gest_types[q_gest_word_idx]
        else:
            q_word, q_type, start, end = q_gest_bounds[0]

        query_gest_bounds[q_gest_word_idx] = (q_word, q_type, start, end)

    # check the gesture boundary thing

    # order the keys of the query gesture bounds such that they are in the order of the text
    query_gest_bounds = {k: v for k, v in enumerate(query_gest_bounds.values())}

    # reassign the query gesture types and words because of the merging and filtering
    # TODO: might be superfluous
    q_idxs = sorted(list(query_gest_bounds.keys()))
    query_gest_types = [query_gest_bounds[i][1] for i in q_idxs]
    query_gest_words = [query_gest_bounds[i][0] for i in q_idxs]

    query_gest_prominence = map_conns_to_prominence(query_gest_words, prominence)

    for i, q in enumerate(q_idxs):
        if query_gest_prominence[i] is None:
            query_gest_prominence[q] = None
            continue
        query_gest_prominence[q] = (query_gest_types[i], *query_gest_prominence[i])

    # gest_type_smps = {i: [] for i in range(len(query_gest_types))}
    for q_gest_type_idx, (q_gest_type, q_gest_word) in enumerate(
        zip(query_gest_types, query_gest_words)
    ):
        smp_2_score = {}
        smp_2_relevant_bounds = {}
        for db_smp_idx, smp_gesture_labels in db_idx_2_gesture_labels.items():

            smp_2_score[db_smp_idx] = 0
            smp_spk = smp_gesture_labels[0]
            smp_gesture_labels = smp_gesture_labels[1:]

            db_smp_prominence = db_idx_2_prominence[db_smp_idx]

            if len(smp_gesture_labels) == 0:
                continue
            
            
            fitered_smp_gesture_labels = []
            filtered_dp_prominence = []
            for gi, g in enumerate(smp_gesture_labels):
                if g["name"] == "beat":
                    continue
                fitered_smp_gesture_labels.append(g)
                filtered_dp_prominence.append(db_smp_prominence[gi])

            smp_gesture_labels = fitered_smp_gesture_labels
            db_smp_prominence = filtered_dp_prominence # it is a list now
                


            smp_gest_types = [s["name"] for s in smp_gesture_labels]
            smp_gest_words = [s["word"] for s in smp_gesture_labels]

            if len(smp_gest_types) == 0:
                continue

            
            # if len(db_smp_prominence) != len(smp_gest_types):
            #     breakpoint()
            assert len(db_smp_prominence) == len(smp_gest_types), f"{len(db_smp_prominence)} != {len(smp_gest_types)}"

            # print(db_smp_prominence)
            smp_prominence = {}
            for sen_idx, conn2val in enumerate(db_smp_prominence):
                if conn2val is None:
                    smp_prominence[sen_idx] = None
                    continue
                smp_conn, prom_val = conn2val
                # if "." in smp_conn:
                #     breakpoint()
                test_conn = smp_gest_words[sen_idx]
                test_conn = "".join(
                    [c for c in test_conn if c.isalnum() or c.isspace()]
                )
                # smp_conn = "".join([c for c in smp_conn if c.isalnum() or c.isspace()])
                # if "".join([c for c in smp_conn if c.isalnum() or c.isspace()]) != test_conn:
                #     breakpoint()
                smp_prominence[sen_idx] = (smp_gest_types[sen_idx], smp_conn, prom_val)

            
            if len(smp_prominence) == 0:
                continue
            # breakpoint()
            # score calculation legend:
            # 2 - type match
            # 4 - text match
            # 5 - speaker match + sense match
            # 7 - speaker match + text match

            if q_gest_type in smp_gest_types:
                smp_2_score[db_smp_idx] += 2

                # get the index of the relevant type
                relevant_smp_typeidx = [
                    typ_idx
                    for typ_idx, typ in enumerate(smp_gest_types)
                    if typ == q_gest_type
                ]

                relevant_smp_words = [
                    smp_gest_words[typ_idx] for typ_idx in relevant_smp_typeidx
                ]

                if smp_spk == speaker_id:
                    smp_2_score[db_smp_idx] += 1

                if q_gest_word in relevant_smp_words:
                    smp_2_score[db_smp_idx] += 5
                    top_rel_idx = relevant_smp_typeidx[
                        relevant_smp_words.index(q_gest_word)
                    ]
                else:
                    

                    sim_list = [
                        get_word_similarity_score(smp_word, q_gest_word)
                        for smp_word in relevant_smp_words
                    ]

                    top_rel_idx = relevant_smp_typeidx[np.argmax(sim_list)]
                    smp_2_score[db_smp_idx] += 3 / (
                        1 + 2 * sim_list[np.argmax(sim_list)]
                    )

                # Prominence score calculation
                # if there are multiple matches of the disco sense in the sample
                # then the prominence values should be compared with the prominence
                # values of the disco sense
                # breakpoint()
                sum_prom_diff = 0
                prom_diff_count = 0
                senidx_2_prom = {}
                # breakpoint()
                for typ_idx in relevant_smp_typeidx:
                    if (
                        smp_prominence[typ_idx] is None
                        or query_gest_prominence[q_gest_type_idx] is None
                    ):
                        continue

                    smp_type, smp_word, smp_prom = smp_prominence[typ_idx]
                    
                    if smp_type != query_gest_prominence[q_gest_type_idx][0]:
                        continue
                    assert smp_type == query_gest_prominence[q_gest_type_idx][0]
                    prom_diff = abs(smp_prom - query_gest_prominence[q_gest_type_idx][-1])

                    senidx_2_prom[typ_idx] = prom_diff
                    sum_prom_diff += 4 / (1 + 2 * prom_diff)
                    prom_diff_count += 1

                if prom_diff_count > 0:
                    prom_diff_score = sum_prom_diff / prom_diff_count
                    smp_2_score[db_smp_idx] += prom_diff_score
                    # sort the relevant_smp_senidx based on the prominence difference
                    # lower the senidx_2_prom value, higher the rank (ascending order)
                    # breakpoint()
                    sorted_senidx = sorted(senidx_2_prom, key=senidx_2_prom.get)

                    if top_rel_idx != sorted_senidx[0]:
                        top_rel_idx = sorted_senidx[0]

                # breakpoint()

                smp_2_relevant_bounds[db_smp_idx] = smp_gesture_labels[top_rel_idx]

        sorted_db_indexes = sorted(smp_2_score, key=smp_2_score.get, reverse=True)
        # breakpoint()

        same_score_tiers = {}
        for smp_idx in sorted_db_indexes:
            if smp_2_score[smp_idx] not in same_score_tiers:
                same_score_tiers[smp_2_score[smp_idx]] = []

            if smp_2_score[smp_idx] > 0:
                same_score_tiers[smp_2_score[smp_idx]].append(smp_idx)

        # breakpoint()
        sorted_db_indexes = []
        for score in sorted(same_score_tiers.keys(), reverse=True):
            tier = same_score_tiers[score]
            if len(tier) > 1:
                tier = sort_sidx_by_textsimilarity(
                    tier, text, encoded_text, text_feat_cache
                )

            sorted_db_indexes += tier
            if len(sorted_db_indexes) >= 10:
                break

        

        # -----------------------------------------
        # get the bounds of the retrieved samples
        d_bounds[q_gest_type_idx] = {}
        sample_indexes[q_gest_type_idx] = sorted_db_indexes[:10]
        for retr_idx in sorted_db_indexes[:10]:
            # if retr_idx not in smp_2_relevant_bounds:
            #     breakpoint()  # should not reach here

            smp_bounds = smp_2_relevant_bounds[retr_idx]
            gest_type = smp_bounds["name"]
            gest_word = smp_bounds["word"]
            gest_start = smp_bounds["start"]
            gest_end = smp_bounds["end"]

            d_bounds[q_gest_type_idx][retr_idx] = (
                gest_word,
                gest_type,
                round(gest_start, 3),
                round(gest_end, 3),
            )
        assert (
            len(d_bounds[q_gest_type_idx]) == len(sample_indexes[q_gest_type_idx]) #== 10
        )

    assert len(d_bounds) == len(sample_indexes) == len(query_gest_bounds)
    return sample_indexes, d_bounds, query_gest_bounds
