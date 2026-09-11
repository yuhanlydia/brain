import math
import pytest

from brain_npp.nsd_evaluation import (
    caption_scores, image_cluster_bootstrap, load_evaluation_cohort, normalize_vqa_answer,
    score_candidate_posterior, score_vqa_predictions,
)


def test_vqa_normalization_and_category_macro_are_symmetric():
    assert normalize_vqa_answer("The two, dogs!") == "2 dogs"
    assert normalize_vqa_answer("dont") == normalize_vqa_answer("don't")
    assert normalize_vqa_answer("1,000") == "1000"
    assert normalize_vqa_answer("1.5.") == "1.5"
    assert normalize_vqa_answer("none") == normalize_vqa_answer("zero") == "0"
    assert normalize_vqa_answer("its") == "its" and normalize_vqa_answer("it's") == "it's"
    # Released processor makes every punctuation decision from the original text.
    assert normalize_vqa_answer("one, cat? dog") == "1 cat dog"
    rows = score_vqa_predictions([
        {"example_id":"1", "image_id":"a", "category":"count", "prediction":"Two", "references":["2"]},
        {"example_id":"2", "image_id":"b", "category":"count", "prediction":"1", "references":["two"]},
        {"example_id":"3", "image_id":"c", "category":"color", "prediction":"Blue.", "references":["blue"]},
    ])
    assert [r["exact_match"] for r in rows["per_example"]] == [1.0, 0.0, 1.0]
    assert rows["accuracy"] == 2/3 and rows["category_macro"] == .75

@pytest.mark.parametrize(("raw","released_output"),[
    ("None; zero!","0 0"),("its / it's","its it's"),("1,000.50?","1000.50"),
    ("couldnt've [the] cats","couldn't've cats"),
])
def test_vqa_normalizer_matches_released_python_evaluation_tools_fixture(raw,released_output):
    assert normalize_vqa_answer(raw)==released_output


def test_bootstrap_joins_identical_image_clusters_and_reports_unmatched():
    result = image_cluster_bootstrap(
        [{"image_id":"a","question_id":1,"source_trial_ids":["t1"],"score":1.0}],
        [{"image_id":"a","question_id":1,"source_trial_ids":["t1"],"score":0.0}],seed=3,samples=50)
    assert result["matched_images"] == result["matched_examples"] == 1 and result["difference"] == 1.0


def test_candidate_scores_report_coverage_entropy_and_conditional_metrics():
    rows=score_candidate_posterior([
        {"example_id":"a","image_id":"i1","candidate_ids":["i1","x"],"probabilities":[.8,.2]},
        {"example_id":"b","image_id":"i2","candidate_ids":["x","y"],"probabilities":[.5,.5]},
    ],ece_bins=2)
    assert rows["coverage"]==.5 and rows["conditional_count"]==1
    assert rows["conditional_nll"]==pytest.approx(-math.log(.8))
    assert rows["conditional_brier"]==pytest.approx(.08)
    assert rows["per_example"][1]["true_target_nll"] is None


def test_caption_scorer_keeps_undefined_spice_subsets_null(monkeypatch):
    class Fake:
        def compute_score(self, refs, hyps):
            return {"All":{"f":.4}, "Objects":{"p":float("nan"), "r":float("nan")}}, [
                {"All":{"f":.4}, "Objects":{"p":float("nan"), "r":float("nan")}}
            ]
    scores = caption_scores({1:["a cat"]}, {1:["cat"]}, spice_scorer=Fake(), standard_scorers=[])
    assert scores["SPICE"] == .4 and scores["SPICE_object_precision"] is None
    assert scores["SPICE_object_precision_defined_count"] == 0 and scores["per_example"][0]["SPICE_object_precision"] is None
    assert scores["per_example"][0]["SPICE"] == .4

def test_candidate_ece_and_zero_probability():
    rows=[{"image_id":"t","candidate_ids":["t","x"] if i<4 else ["x","t"],"probabilities":[.8,.2]} for i in range(5)]
    assert score_candidate_posterior(rows)["conditional_ece"] == pytest.approx(0)
    zero=score_candidate_posterior([{"image_id":"t","candidate_ids":["t","x"],"probabilities":[0,1]}])
    assert zero["conditional_nll"] is None and zero["conditional_nll_infinite_count"]==1


def test_evaluation_cohort_validates_hashes_metadata_and_repeat_order(tmp_path):
    import hashlib, json
    manifest = tmp_path / "manifest"; manifest.write_text("m")
    vqa = tmp_path / "vqa"; vqa.write_text("v")
    caption = tmp_path / "caption"; caption.write_text("c")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    payload = {"schema_version":1, "input_sha256":{"manifest":sha(manifest), "vqa_test":sha(vqa), "caption_test":sha(caption)},
               "examples":[{"image_id":"nsd00001", "question_id":2, "answer_id":3, "category":"color",
                            "trial_ids_by_subject":{"subj01":["r0", "r1", "r2"]}}]}
    path = tmp_path / "cohort.json"; path.write_text(json.dumps(payload))
    cohort = load_evaluation_cohort(path, manifest_path=manifest, vqa_path=vqa, caption_path=caption,
                                    valid_trials={"r0":("subj01","nsd00001",0), "r1":("subj01","nsd00001",1), "r2":("subj01","nsd00001",2)})
    assert cohort.for_subject("subj01", repeat_prefix=2)[0].trial_ids == ("r0", "r1")
    payload["examples"][0]["trial_ids_by_subject"]["subj01"][1] = "r2"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="repeat"):
        load_evaluation_cohort(path, manifest_path=manifest, vqa_path=vqa, caption_path=caption,
                               valid_trials={"r0":("subj01","nsd00001",0), "r1":("subj01","nsd00001",1), "r2":("subj01","nsd00001",2)})
