import json
import os
from multiwoz_config import args, multiwoz_resource_folder


def evaluate(all_prediction, name, slot_temp):
    if args["genSample"]:
        json.dump(all_prediction, open(os.path.join(multiwoz_resource_folder, "all_prediction_{}.json"
                                                    .format(name)), 'w'), indent=4)

    joint_acc_score_ptr, f1_score_ptr, turn_acc_score_ptr = evaluate_metrics(all_prediction, "pred_bs_ptr",
                                                                             slot_temp)

    evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                          "Joint F1": f1_score_ptr}
    print(evaluation_metrics)
    joint_acc_score = joint_acc_score_ptr  # (joint_acc_score_ptr + joint_acc_score_class)/2

    return joint_acc_score, f1_score_ptr


def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, f1_pred, f1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[t]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
            f1_pred += temp_f1
            f1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    f1_score = f1_pred / float(f1_count) if f1_count != 0 else 0
    return joint_acc_score, f1_score, turn_acc_score


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    acc_total = len(slot_temp)
    acc = len(slot_temp) - miss_gold - wrong_pred
    acc = acc / float(acc_total)
    return acc


def compute_prf(gold, pred):
    tp, fp, fn = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                tp += 1
            else:
                fn += 1
        for p in pred:
            if p not in gold:
                fp += 1
        precision = tp / float(tp + fp) if (tp + fp) != 0 else 0
        recall = tp / float(tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, f1, count = 1, 1, 1, 1
        else:
            precision, recall, f1, count = 0, 0, 0, 1
    return f1, recall, precision, count
