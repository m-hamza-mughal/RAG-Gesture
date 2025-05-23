import torch
import numpy as np
import random

from .utils import sort_sidx_by_textsimilarity, get_word_similarity_score


def gesture_type_retrieval(
    text,
    gesture_labels,
    speaker_id,
    db_idx_2_gesture_labels,
    encoded_text,
    text_feat_cache,
):
    """
    retrieval function for semantic gesture labels
    """
    # keep the gesture labels which are not beat
    gesture_labels = [g for g in gesture_labels if g["name"] != "beat"]

    d_bounds = {}
    sample_indexes = {}
    query_gest_bounds = {}

    if len(gesture_labels) == 0:
        return sample_indexes, d_bounds, query_gest_bounds
    else:
        # breakpoint()
        query_gest_types = [g["name"] for g in gesture_labels]
        # query_gest_bounds = [(g["start"], g["end"]) for g in gesture_labels]
        query_gest_words = [g["word"] for g in gesture_labels]

        query_gest_bounds = {
            q_idx: (g["word"].lower(), g["name"], g["start"], g["end"])
            for q_idx, g in enumerate(gesture_labels)
        }

        # gest_type_smps = {i: [] for i in range(len(query_gest_types))}
        for q_gest_type_idx, (q_gest_type, q_gest_word) in enumerate(
            zip(query_gest_types, query_gest_words)
        ):
            # breakpoint()
            smp_2_score = {}
            smp_2_relevant_bounds = {}
            # breakpoint() # check len of db_idx_2_gesture_labels
            for db_smp_idx, smp_gesture_labels in db_idx_2_gesture_labels.items():
                
                # # skip 50% of the samples
                # if random.random() < 0.5:
                #     continue

                smp_2_score[db_smp_idx] = 0
                smp_spk = smp_gesture_labels[0]
                smp_gesture_labels = smp_gesture_labels[1:]
                smp_gesture_labels = [
                    g for g in smp_gesture_labels if g["name"] != "beat"
                ]

                smp_gest_types = [s["name"] for s in smp_gesture_labels]
                smp_gest_words = [s["word"] for s in smp_gesture_labels]

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
                        smp_2_score[db_smp_idx] += 2

                    if q_gest_word in relevant_smp_words:
                        smp_2_score[db_smp_idx] += 5
                        top_rel_idx = relevant_smp_typeidx[
                            relevant_smp_words.index(q_gest_word)
                        ]
                    else:
                        # pass
                        # breakpoint()  # check the word similarity calculation
                        # word similarity
                        # q_word_f = text_feat_extractor.extract_feature(q_gest_word)
                        # smp_word_fs = [
                        #     text_feat_extractor.extract_feature(smp_word)
                        #     for smp_word in relevant_smp_words
                        # ]
                        # # similarity list
                        # sim_list = [
                        #     torch.mm(q_word_f, smp_word_f.T).diagonal().mean().item()
                        #     for smp_word_f in smp_word_fs
                        # ]
                        sim_list = [
                            get_word_similarity_score(smp_word, q_gest_word)
                            for smp_word in relevant_smp_words
                        ]

                        top_rel_idx = relevant_smp_typeidx[np.argmax(sim_list)]
                        smp_2_score[db_smp_idx] += 3 / (
                            1 + 2 * sim_list[np.argmax(sim_list)]
                        )

                    smp_2_relevant_bounds[db_smp_idx] = smp_gesture_labels[top_rel_idx]

            sorted_db_indexes = sorted(smp_2_score, key=smp_2_score.get, reverse=True)
            # print(
            #     smp_2_score[sorted_db_indexes[0]],
            #     smp_2_score[sorted_db_indexes[1]],
            #     smp_2_score[sorted_db_indexes[2]],
            # )

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

            # breakpoint()

            # -----------------------------------------
            # get the bounds of the retrieved samples
            d_bounds[q_gest_type_idx] = {}
            sample_indexes[q_gest_type_idx] = sorted_db_indexes[:10]
            for retr_idx in sorted_db_indexes[:10]:
                if retr_idx not in smp_2_relevant_bounds:
                    breakpoint()  # should not reach here

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
                len(d_bounds[q_gest_type_idx])
                == len(sample_indexes[q_gest_type_idx])
                # == 10
            )
            # print(len(d_bounds[q_gest_type_idx]))

    assert len(d_bounds) == len(sample_indexes) == len(query_gest_bounds)
    return sample_indexes, d_bounds, query_gest_bounds

    #     for smp_idx, smp_gesture_labels in self.idx_2_gesture_labels.items():
    #         smp_spk = smp_gesture_labels[0]
    #         smp_gesture_labels = smp_gesture_labels[1:]
    #         smp_gest_types = [s["name"] for s in smp_gesture_labels]

    #         for smp_gt_idx, smp_gest_type in enumerate(smp_gest_types):
    #             if smp_gest_type == "beat":
    #                 continue
    #             if smp_gest_type in query_gest_types:
    #                 gest_type_smps[query_gest_types.index(smp_gest_type)].append(
    #                     smp_idx
    #                 )  # smp_gt_idx

    #     if len(gest_type_smps) == 0:
    #         breakpoint()

    #     # breakpoint()  # check the q_gest_type_idx
    #     selected_text_features = {}
    #     for q_gest_type_idx, smp_indices in gest_type_smps.items():

    #         for db_smp_idx in smp_indices:
    #             smp_text_f, smp_spk = self.idx_2_text[db_smp_idx]
    #             smp_text_f_norm = smp_text_f[0] / smp_text_f[0].norm(
    #                 dim=1, keepdim=True
    #             )
    #             sim = torch.mm(text_feature, smp_text_f_norm.T).diagonal().mean().item()

    #             selected_text_features[db_smp_idx] = sim

    #         # reorder the smp_indices based on the text similarity
    #         sorted_smp_indexes = sorted(
    #             selected_text_features,
    #             key=selected_text_features.get,
    #             reverse=True,
    #         )

    #         q_gest_type = query_gest_types[q_gest_type_idx]
    #         q_gest_bound = query_gest_bounds[q_gest_type_idx]

    #         relevant_gest_bounds = []
    #         relevant_gest_types = []
    #         for smp_idx in sorted_smp_indexes:

    #             smp_gesture_labels = self.idx_2_gesture_labels[smp_idx]
    #             smp_spk = smp_gesture_labels[0]
    #             smp_gesture_labels = smp_gesture_labels[1:]
    #             smp_gest_types = [s["name"] for s in smp_gesture_labels]
    #             if q_gest_type not in smp_gest_types:
    #                 continue

    #             # breakpoint()  # check word in gest bounds
    #             smp_gest_bounds = [
    #                 (
    #                     s["start"],
    #                     s["end"],
    #                     s["word"] if s["word"] != np.nan else "",
    #                 )
    #                 for s in smp_gesture_labels
    #             ]
    #             # breakpoint()

    #             relevant_gest_type = smp_gest_types[smp_gest_types.index(q_gest_type)]
    #             relevant_gest_types.append(relevant_gest_type)
    #             relevant_gest_bound = smp_gest_bounds[smp_gest_types.index(q_gest_type)]
    #             relevant_gest_bounds.append(relevant_gest_bound)

    #             if len(relevant_gest_types) == self.num_retrieval:
    #                 break

    #         d_bounds[q_gest_type_idx] = {
    #             "sample_indexes": sorted_smp_indexes[: self.num_retrieval],
    #             "relevant_gest_types": relevant_gest_types,
    #             "relevant_gest_bounds": relevant_gest_bounds,
    #         }

    # return d_bounds
