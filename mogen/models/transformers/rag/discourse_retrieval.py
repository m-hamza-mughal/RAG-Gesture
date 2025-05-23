import torch
import copy
import random

from .utils import sort_sidx_by_textsimilarity, map_conns_to_prominence


def discourse_retrieval(
    text,
    discourse,
    prominence,
    speaker_id,
    # length,
    db_idx_2_sense,
    db_idx_2_discbounds,
    db_idx_2_prominence,
    encoded_text,
    text_feat_cache,
):
    """
    discourse retrieval

    Args:
    text: text in query
    discourse: discourse in query
    prominence: prosodic prominence values of the query
                - list of tuples (word, start, end, prominence)
    speaker_id: speaker_id in query
    db_idx_2_sense: mapping of db index to speaker_id of db_sample, list of (sense, text)
                    which the sample contains
    db_idx_2_discbounds: mapping of db index to discourse bounds of db_sample
    db_idx_2_prominence: mapping of db index to prominence tuples of db_sample
    text_feat_extractor: text feature extractor
    text_feat_cache: text feature cache
    """
    # breakpoint()
    d_bounds = {}
    sample_indexes = {}
    disco_query_bounds = {}
    if len(discourse) == 0:
        return sample_indexes, d_bounds, disco_query_bounds
    else:
        # start = time.time()

        # query example 1 - sense 0 - text
        disco_senses = [d[1] for d in discourse]
        disco_conns = [d[0] for d in discourse]  # .lower()

        disco_query_bounds = {
            d_idx: (d[0].lower(), d[1], d[6], d[7]) for d_idx, d in enumerate(discourse)
        }

        disco_prominence = map_conns_to_prominence(disco_conns, prominence)

        # breakpoint()  # check the prominence values and word and the sense ordering
        # following code

        # sense to prominence mapping
        for disc_idx, conn2val in disco_prominence.items():
            if conn2val is None:
                continue

            disc_conn, prom_val = conn2val
            test_conn = disco_conns[disc_idx]
            test_conn = "".join([c for c in test_conn if c.isalnum() or c.isspace()])
            assert disc_conn == test_conn, f"{disc_conn} != {test_conn}"
            disco_prominence[disc_idx] = (disco_senses[disc_idx], prom_val)
        # disco_prominence = {
        #     sen_idx: (disco_sen, disco_prom[1])
        #     for sen_idx, (disco_sen, disco_prom) in enumerate(
        #         zip(disco_senses, relevant_dps)
        #     )
        # }

        for disco_idx, (disco_sense, disco_text) in enumerate(
            zip(disco_senses, disco_conns)
        ):

            # filter based on sense of all disco entries instead of first one
            # and rank based on similarity of all disco entries
            smp_2_score = {}
            smp_2_relevantbounds = {}
            # breakpoint() # check len of db_idx_2_sense

            
            for smp_idx, smp_disco_senses in db_idx_2_sense.items():
                
                # # skip 50% of the samples
                # if random.random() < 0.5:
                #     continue


                smp_2_score[smp_idx] = 0
                smp_spk = smp_disco_senses[0]  # speaker id
                smp_disco = smp_disco_senses[1:]  # list of (sense, text)

                if len(smp_disco) == 0:
                    continue

                # list of (word, prominence)
                db_smp_prominence = db_idx_2_prominence[smp_idx]
                # breakpoint()

                # retrival example 0 - sense 1 - text
                smp_senses = [d[0] for d in smp_disco]
                smp_conns = [d[1] for d in smp_disco]  # .lower()
                # sense to prominence mapping # same length as senses
                assert len(db_smp_prominence) == len(
                    smp_senses
                ), f"{len(db_smp_prominence)} != {len(smp_senses)}"
                smp_prominence = {}
                for sen_idx, conn2val in db_smp_prominence.items():
                    if conn2val is None:
                        smp_prominence[sen_idx] = None
                        continue
                    smp_conn, prom_val = conn2val
                    if "." in smp_conn:
                        breakpoint()
                    test_conn = smp_conns[sen_idx]
                    test_conn = "".join(
                        [c for c in test_conn if c.isalnum() or c.isspace()]
                    )
                    if smp_conn != test_conn:
                        breakpoint()
                    smp_prominence[sen_idx] = (smp_senses[sen_idx], prom_val)

                # smp_prominence = {
                #     sen_idx: (smp_sen, smp_prom[1])
                #     for sen_idx, (smp_sen, smp_prom) in enumerate(
                #         zip(smp_senses, smp_prominence.values())
                #     )
                # }

                # score calculation legend:
                # 2 - sense match
                # 4 - text match
                # 5 - speaker match + sense match
                # 7 - speaker match + text match

                if disco_sense in smp_senses:
                    # breakpoint()
                    smp_2_score[smp_idx] += 2

                    relevant_smp_senidx = [
                        sen_idx
                        for sen_idx, smp_sen in enumerate(smp_senses)
                        if smp_sen == disco_sense
                    ]
                    top_rel_idx = relevant_smp_senidx[0]
                    top_rel_chosen = False
                    relevant_smp_conns = [
                        smp_conns[sen_idx] for sen_idx in relevant_smp_senidx
                    ]
                    # TODO: should we normalize the connective text before comparison?
                    # relevant_smp_proms = [
                    #     smp_prominence[sen_idx] for sen_idx in relevant_smp_senidx
                    # ]

                    if disco_text in relevant_smp_conns:
                        smp_2_score[smp_idx] += 4
                        top_rel_idx = relevant_smp_senidx[
                            relevant_smp_conns.index(disco_text)
                        ]
                        top_rel_chosen = True

                    if smp_spk == speaker_id:
                        smp_2_score[smp_idx] += 3

                    # if there are multiple matches of the disco sense in the sample
                    # then the prominence values should be compared with the prominence
                    # values of the disco sense
                    sum_prom_diff = 0
                    prom_diff_count = 0
                    senidx_2_prom = {}
                    for sen_idx in relevant_smp_senidx:
                        if (
                            smp_prominence[sen_idx] is None
                            or disco_prominence[disco_idx] is None
                        ):
                            continue

                        smp_sen, smp_prom = smp_prominence[sen_idx]
                        assert smp_sen == disco_prominence[disco_idx][0]
                        prom_diff = abs(smp_prom - disco_prominence[disco_idx][1])

                        senidx_2_prom[sen_idx] = prom_diff
                        sum_prom_diff += 4 / (1 + 2 * prom_diff)
                        prom_diff_count += 1

                    if prom_diff_count > 0:
                        prom_diff_score = sum_prom_diff / prom_diff_count
                        smp_2_score[smp_idx] += prom_diff_score
                        # sort the relevant_smp_senidx based on the prominence difference
                        # lower the senidx_2_prom value, higher the rank (ascending order)
                        # breakpoint()
                        sorted_senidx = sorted(senidx_2_prom, key=senidx_2_prom.get)

                        if top_rel_idx != sorted_senidx[0] and not top_rel_chosen:
                            top_rel_idx = sorted_senidx[0]

                        # assign the top ranked senidx to the smp_2_relevantbounds
                    smp_2_relevantbounds[smp_idx] = db_idx_2_discbounds[smp_idx][
                        top_rel_idx
                    ]

                    # -----------------------------------------

                    # if disco_sense in smp_prominence:
                    #     # here the prominence score of smp_prominence[disco_sense] should
                    #     # be compared with disco_prominence[disco_sense]
                    #     prom_diff = abs(
                    #         smp_prominence[disco_sense] - disco_prominence[disco_sense]
                    #     )
                    #     # higher the difference, lower the score
                    #     smp_2_score[smp_idx] += 5 / (1 + 2 * prom_diff)

                # # we skip the db sample since it does not contain the sense of query
                # if smp_2_score[smp_idx] == 0:
                #     continue

            # breakpoint()
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
            d_bounds[disco_idx] = {}
            # breakpoint()
            sample_indexes[disco_idx] = sorted_db_indexes[:10]
            for retr_idx in sorted_db_indexes[:10]:
                if retr_idx not in smp_2_relevantbounds:
                    breakpoint()  # should not reach here

                smp_bounds = smp_2_relevantbounds[retr_idx]

                conn, sense, word_start, word_end = (
                    smp_bounds[1],
                    smp_bounds[0],
                    smp_bounds[4],
                    smp_bounds[5],
                )
                d_bounds[disco_idx][retr_idx] = (
                    conn,
                    sense,
                    round(word_start, 3),
                    round(word_end, 3),
                )

            assert len(d_bounds[disco_idx]) == len(sample_indexes[disco_idx]) #== 10
            # print(sample_indexes[disco_idx])

            # # did_2_rid = {}
            # r_senses = [rs[0] for rs in db_idx_2_discbounds[retr_idx]]

            # # prev_match = -1
            # if disco_sense not in r_senses:
            #     breakpoint() # should not reach here.
            #     d_bounds[disco_idx][retr_idx] = []
            #     # did_2_rid[d_i] = -1  # no match
            #     # breakpoint()
            #     continue

            # matches = [ri for ri, rs in enumerate(r_senses) if rs == d_sense]
            # if len(matches) == 1:
            #     did_2_rid[d_i] = matches[0]
            #     prev_match = matches[0]
            # else:
            #     filtered_matches = [m for m in matches if m > prev_match]

            #     # breakpoint()
            #     if len(filtered_matches) == 0:
            #         # breakpoint()
            #         filtered_matches = matches

            #     did_2_rid[d_i] = filtered_matches[0]
            #     prev_match = filtered_matches[0]

            # # get bounds
            # r_bounds = db_idx_2_discbounds[retr_idx][did_2_rid[d_i]]  # [1:]
            # breakpoint()  # TODO: format the bounds correctly.
            # d_bounds[retr_idx][d_i] = r_bounds
            # breakpoint()

        # breakpoint()

        # sample_indexes = sorted_db_indexes

    assert len(d_bounds) == len(sample_indexes) == len(disco_query_bounds)
    return sample_indexes, d_bounds, disco_query_bounds
