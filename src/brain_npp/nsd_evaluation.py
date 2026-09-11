"""Per-example VQA/caption metrics and paired image-cluster summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from typing import Mapping, Sequence


_ARTICLES = {"a", "an", "the"}
_DIGITS = {"none":"0", "zero":"0", "one":"1", "two":"2", "three":"3", "four":"4", "five":"5",
           "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10"}
_CONTRACTIONS = {
 "aint":"ain't","arent":"aren't","cant":"can't","couldnt":"couldn't","didnt":"didn't",
 "doesnt":"doesn't","dont":"don't","hadnt":"hadn't","hasnt":"hasn't","havent":"haven't",
 "hes":"he's","im":"i'm","isnt":"isn't","itll":"it'll","ive":"i've",
 "shouldnt":"shouldn't","theyre":"they're","wasnt":"wasn't","werent":"weren't",
 "wont":"won't","wouldnt":"wouldn't","youre":"you're","youve":"you've",
 "couldve":"could've","couldn'tve":"couldn't've","couldnt've":"couldn't've","hadnt've":"hadn't've","hadn'tve":"hadn't've",
 "hed":"he'd","hed've":"he'd've","he'dve":"he'd've","howd":"how'd","howll":"how'll","hows":"how's",
 "itd":"it'd","itd've":"it'd've","it'dve":"it'd've","maam":"ma'am","mightnt":"mightn't","mightve":"might've",
 "mustnt":"mustn't","mustve":"must've","neednt":"needn't","notve":"not've","oclock":"o'clock","oughtnt":"oughtn't",
 "shant":"shan't","shed've":"she'd've","she'dve":"she'd've","shouldve":"should've","shouldnt've":"shouldn't've",
 "somebodyd've":"somebody'd've","somebodyll":"somebody'll","somebodys":"somebody's","someoned":"someone'd",
 "someonell":"someone'll","someones":"someone's","somethingd":"something'd","somethingll":"something'll",
 "thats":"that's","thered":"there'd","therere":"there're","theres":"there's","theyd":"they'd","theyll":"they'll","theyve":"they've",
 "twas":"'twas","weve":"we've","whatll":"what'll","whatre":"what're","whats":"what's","whatve":"what've",
 "whens":"when's","whered":"where'd","wheres":"where's","whereve":"where've","whod":"who'd","wholl":"who'll",
 "whos":"who's","whove":"who've","whyll":"why'll","whyre":"why're","whys":"why's","wouldve":"would've",
 "yall":"y'all","yall'll":"y'all'll","youd":"you'd","youll":"you'll",
 "mightnt've":"mightn't've","mightn'tve":"mightn't've","shouldn'tve":"shouldn't've",
 "somebody'd":"somebodyd","somebody'dve":"somebody'd've","someoned've":"someone'd've","someone'dve":"someone'd've",
 "somethingd've":"something'd've","something'dve":"something'd've","thered've":"there'd've","there'dve":"there'd've",
 "theyd've":"they'd've","they'dve":"they'd've","wed've":"we'd've","we'dve":"we'd've",
 "whod've":"who'd've","who'dve":"who'd've","wouldnt've":"wouldn't've","wouldn'tve":"wouldn't've",
 "yall'll":"y'all'll","y'allll":"y'all'll","yall'd've":"y'all'd've","y'alld've":"y'all'd've","y'all'dve":"y'all'd've",
 "youd've":"you'd've","you'dve":"you'd've","ow's'at":"'ow's'at","'ows'at":"'ow's'at","'ow'sat":"'ow's'at",
 "she's":"she's","let's":"let's","id've":"i'd've","i'dve":"i'd've",
}
# Port of the released VQA PythonEvaluationTools vqaEval.py answer processor.
VQA_NORMALIZATION_POLICY = "VQA-PythonEvaluationTools-vqaEval.py:processPunctuation+processDigitArticle:v2"
_PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


@dataclass(frozen=True)
class CohortExample:
    image_id: str
    question_id: int
    answer_id: int
    category: str
    trial_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCohort:
    examples: tuple[Mapping, ...]
    sha256: str

    def for_subject(self, subject_id: str, *, repeat_prefix: int = 1) -> tuple[CohortExample, ...]:
        if repeat_prefix not in (1, 2, 3):
            raise ValueError("repeat_prefix must be 1, 2, or 3")
        result = []
        for row in self.examples:
            trials = row["trial_ids_by_subject"].get(subject_id)
            if trials is None or len(trials) < repeat_prefix:
                raise ValueError(f"cohort missing {subject_id} repeat prefix")
            result.append(CohortExample(row["image_id"], int(row["question_id"]), int(row["answer_id"]),
                                        row["category"], tuple(trials[:repeat_prefix])))
        return tuple(result)


def load_evaluation_cohort(path, *, manifest_path, vqa_path, caption_path, valid_trials: Mapping[str, tuple[str, str, int]]):
    path = __import__("pathlib").Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("examples"), list):
        raise ValueError("invalid evaluation cohort schema")
    sha = lambda p: hashlib.sha256(__import__("pathlib").Path(p).read_bytes()).hexdigest()
    expected = {"manifest": sha(manifest_path), "vqa_test": sha(vqa_path), "caption_test": sha(caption_path)}
    inputs = payload.get("input_sha256", {})
    # Accept descriptive keys used by preparation scripts while still requiring exact values.
    aliases = {"manifest": "nsd_manifest.jsonl", "vqa_test": "vqa_test.jsonl", "caption_test": "captions_test.jsonl"}
    for name, digest in expected.items():
        candidates = {inputs.get(name), inputs.get(name + "_sha256"), inputs.get(aliases[name])}
        if digest not in candidates:
            raise ValueError(f"evaluation cohort {name} hash mismatch")
    seen = set()
    for row in payload["examples"]:
        identity = (row.get("image_id"), row.get("question_id"), row.get("answer_id"))
        if identity in seen or not isinstance(row.get("category"), str):
            raise ValueError("evaluation cohort metadata is invalid or duplicated")
        seen.add(identity)
        for subject, trials in row.get("trial_ids_by_subject", {}).items():
            for repeat, trial_id in enumerate(trials):
                if valid_trials.get(trial_id) != (subject, row["image_id"], repeat):
                    raise ValueError("evaluation cohort trial subject/image/repeat mismatch")
    return EvaluationCohort(tuple(payload["examples"]), sha(path))


def normalize_vqa_answer(value: str) -> str:
    """Symmetric exact-match normalization adapted from VQA PythonEvaluationTools."""
    text = str(value).replace("\n", " ").replace("\t", " ").strip().lower()
    original = text
    for punct in _PUNCT:
        text = text.replace(punct, "" if (punct + " " in original or " " + punct in original or _COMMA_STRIP.search(original)) else " ")
    text = _PERIOD_STRIP.sub("", text)
    words = []
    for word in text.split():
        word = _DIGITS.get(word, word); word = _CONTRACTIONS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)


def score_vqa_predictions(rows: Sequence[Mapping]) -> dict:
    per_example, categories = [], defaultdict(list)
    for row in rows:
        prediction = normalize_vqa_answer(row["prediction"])
        references = [normalize_vqa_answer(x) for x in row["references"]]
        exact = float(prediction in references)
        value = {**dict(row), "normalized_prediction": prediction,
                 "normalized_references": references, "exact_match": exact,
                 "normalization_policy": VQA_NORMALIZATION_POLICY}
        per_example.append(value); categories[str(row["category"])].append(exact)
    if not per_example:
        raise ValueError("VQA evaluation has no examples")
    category_scores = {name: sum(values) / len(values) for name, values in categories.items()}
    return {"accuracy": sum(x["exact_match"] for x in per_example) / len(per_example),
            "category_macro": sum(category_scores.values()) / len(category_scores),
            "category_accuracy": category_scores, "per_example": per_example,
            "reference_semantics": "auto-generated-answer-alternatives-not-human-consensus"}


def image_cluster_bootstrap(left: Sequence[Mapping], right: Sequence[Mapping], *, seed: int, samples: int = 2000) -> dict:
    def keyed(rows):
        out={}
        for row in rows:
            key=(str(row["image_id"]),str(row["question_id"]),tuple(map(str,row["source_trial_ids"])))
            if key in out: raise ValueError("duplicate paired example identity")
            out[key]=float(row["score"])
        return out
    a,b=keyed(left),keyed(right);common=sorted(set(a)&set(b))
    if not common: raise ValueError("paired bootstrap has no common example identities")
    grouped=defaultdict(list)
    for key in common: grouped[key[0]].append(a[key]-b[key])
    differences=[sum(grouped[k])/len(grouped[k]) for k in sorted(grouped)]
    rng = random.Random(seed); draws = []
    for _ in range(samples):
        draws.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    draws.sort()
    quantile = lambda p: draws[min(len(draws)-1, int(p * len(draws)))]
    return {"difference": sum(differences)/len(differences), "ci95": [quantile(.025), quantile(.975)],
            "matched_images": len(grouped), "matched_examples":len(common),
            "unmatched_left_examples":[list(x) for x in sorted(set(a)-set(b))],
            "unmatched_right_examples":[list(x) for x in sorted(set(b)-set(a))], "bootstrap_unit": "image_id"}


def score_candidate_posterior(rows: Sequence[Mapping], *, ece_bins: int = 10) -> dict:
    """Report gallery coverage; target metrics are conditional on genuine support."""
    if not rows or ece_bins < 1: raise ValueError("candidate evaluation requires rows and positive bins")
    per_example=[]; conditional=[]
    for row in rows:
        ids=list(row["candidate_ids"]); probabilities=[float(x) for x in row["probabilities"]]
        if len(ids)!=len(probabilities) or not ids or any(x<0 or not math.isfinite(x) for x in probabilities) or not math.isclose(sum(probabilities),1,rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError("candidate probabilities are invalid")
        entropy=-sum(x*math.log(x) for x in probabilities if x>0); true=row["image_id"]
        value={**dict(row),"entropy":entropy,"coverage":true in ids,"true_target_nll":None,"brier":None,"top_confidence":None,"top_correct":None,"true_target_zero_probability":False}
        if true in ids:
            index=ids.index(true); p=probabilities[index]
            top=max(range(len(ids)),key=probabilities.__getitem__)
            value["true_target_nll"]=-math.log(p) if p>0 else None; value["true_target_zero_probability"]=p==0
            value["top_confidence"]=probabilities[top];value["top_correct"]=float(top==index)
            value["brier"]=sum((x-float(i==index))**2 for i,x in enumerate(probabilities));conditional.append(value)
        per_example.append(value)
    ece=0.0
    if conditional:
        for bin_index in range(ece_bins):
            low,high=bin_index/ece_bins,(bin_index+1)/ece_bins
            selected=[x for x in conditional if low <= x["top_confidence"] and (bin_index==ece_bins-1 or x["top_confidence"] < high)]
            if selected: ece += len(selected)/len(conditional)*abs(sum(x["top_confidence"] for x in selected)/len(selected)-sum(x["top_correct"] for x in selected)/len(selected))
    def mean(key):
        values=[x[key] for x in conditional if x[key] is not None]
        return sum(values)/len(values) if values and len(values)==len(conditional) else None
    return {"coverage":len(conditional)/len(rows),"conditional_count":len(conditional),
            "conditional_nll":mean("true_target_nll"),"conditional_nll_infinite_count":sum(x["true_target_zero_probability"] for x in conditional),"conditional_brier":mean("brier"),
            "conditional_ece":ece if conditional else None,"mean_entropy":sum(x["entropy"] for x in per_example)/len(per_example),
            "per_example":per_example,"conditioning":"sampled-gallery"}


def caption_scores(references: Mapping[int, Sequence[str]], hypotheses: Mapping[int, Sequence[str]], *,
                   spice_scorer=None, standard_scorers=None) -> dict:
    """Run canonical COCO scorers; expose SPICE object P/R as a reference proxy."""
    scorers = list(standard_scorers or [])
    if spice_scorer is None:
        from pycocoevalcap.spice.spice import Spice
        spice_scorer = Spice()
    finite=lambda x: float(x) if math.isfinite(float(x)) else None
    aggregate, per_rows = {}, {key: {"example_id": key} for key in sorted(hypotheses)}
    for scorer, names in scorers:
        score, values = scorer.compute_score(references, hypotheses)
        names = [names] if isinstance(names, str) else names
        score = [score] if not isinstance(score, (list, tuple)) else score
        columns = [values] if len(names) == 1 else values
        for name, total, column in zip(names, score, columns):
            aggregate[name] = finite(total)
            for key, value in zip(sorted(hypotheses), column): per_rows[key][name] = finite(value)
    total, values = spice_scorer.compute_score(references, hypotheses)
    aggregate["SPICE"] = finite(total.get("All", {}).get("f", float("nan")) if isinstance(total, dict) else total)
    defined_all=[];defined_p = []; defined_r = []
    for key, value in zip(sorted(hypotheses), values):
        all_f=finite(value.get("All",{}).get("f",float("nan")));per_rows[key]["SPICE"]=all_f
        if all_f is not None:defined_all.append(all_f)
        objects = value.get("Objects", {})
        for field, target in (("p", defined_p), ("r", defined_r)):
            raw = objects.get(field, float("nan")); cooked = float(raw) if math.isfinite(float(raw)) else None
            per_rows[key]["SPICE_object_" + ("precision" if field == "p" else "recall")] = cooked
            if cooked is not None: target.append(cooked)
    aggregate.update({"SPICE_object_precision": sum(defined_p)/len(defined_p) if defined_p else None,
                      "SPICE_object_recall": sum(defined_r)/len(defined_r) if defined_r else None,
                      "SPICE_defined_count":len(defined_all),"SPICE_object_precision_defined_count":len(defined_p),"SPICE_object_recall_defined_count":len(defined_r),
                      "SPICE_object_denominator": len(values), "per_example": list(per_rows.values()),
                      "SPICE_object_semantics": "reference-grounded-factual-object-proxy"})
    return aggregate
