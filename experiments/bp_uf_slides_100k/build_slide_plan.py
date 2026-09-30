"""Prepare reviewable Google Slides requests; no network calls or deck writes."""
from pathlib import Path
import argparse
import copy
import json


def main(workspace: Path):
    raw = json.loads((workspace / "raw-template.json").read_text())
    deck = raw["structuredContent"]
    exemplar_id = "hd_ler_relative_1m_20260927"
    exemplar = next(s for s in deck["slides"] if s["objectId"] == exemplar_id)
    elements = {e["objectId"].rsplit("_", 1)[-1]: e for e in exemplar["pageElements"]}
    specs = [
        {
            "id": "hd_bp_accuracy_100k_20260929", "figure": "accuracy",
            "role": "Introduce the fixed-budget, single-patch experiment and show all seven accuracy curves.",
            "title": "Fixed BP + UF: LER (100k shots)",
            "heading1": "Single-patch test",
            "body1": "SI1000 p = 0.3%\n4d rounds; ideal boundaries\n100k paired shots per d\nZero yokes",
            "heading2": "Fixed BP pre-decoder",
            "body2": "1, 2, 5 or 10 iterations\nDamping = 0.5\nUF runs on every shot",
            "caption": "LER per full 4d-round memory shot; logical error = either observable wrong.",
            "alt": "Logical error rate (LER) per full memory shot versus distance 7, 9, 11 and 13 for seven decoders, with 95% Wilson intervals. BP10 plus weighted UF has the lowest observed LER at all four distances.",
        },
        {
            "id": "hd_bp_relative_100k_20260929", "figure": "relative_accuracy",
            "role": "Compare LER to correlated MWPM using the same baseline-relative presentation as slide 13.",
            "title": "LER gap versus the baseline (100k shots)",
            "heading1": "BP5 + UF",
            "body1": "−2.2% at d = 7\n+4.2% at d = 13",
            "heading2": "BP10 + UF",
            "body2": "−10.2% at d = 7\n−7.9% at d = 13",
            "caption": "Baseline: correlated MWPM. Gap = 100 × (LER / baseline LER − 1); negative is better.",
            "alt": "Relative LER gap versus correlated MWPM, shown as the zero baseline. BP1, BP2 and correlated UF have higher LER. BP5 ranges from −2.2% at d=7 to +4.2% at d=13; BP10 ranges from −10.2% at d=7 to −7.9% at d=13. Rates use full memory shots.",
        },
        {
            "id": "hd_bp_latency_100k_20260929", "figure": "serial_latency",
            "role": "Make the accuracy versus serial runtime tradeoff explicit for all seven decoders.",
            "title": "Serial decode time grows with BP",
            "heading1": "At d = 13",
            "body1": "BP10 + UF: 234.0 ms\nCorrelated UF: 14.9 ms\nCorr. MWPM: 0.96 ms",
            "heading2": "32-core collection",
            "body2": "OpenMP across native shots\nSerial timing measured on\n1,024 pilot shots",
            "caption": "Current software timings; excludes sampling/setup. MWPM and BP/UF timer scopes differ.",
            "alt": "Log-scale plot of mean single-thread decode time. BP10 plus UF ranges from 28.8 to 234.0 ms over distances 7 to 13; correlated MWPM ranges from 0.087 to 0.961 ms. These are software measurements, not hardware latency estimates.",
        },
        {
            "id": "hd_bp_breakdown_100k_20260929", "figure": "bp_uf_breakdown",
            "role": "Separate BP initialization, BP rounds, graph-weight projection, and UF time.",
            "title": "Where time goes: BP rounds + UF",
            "heading1": "BP10 + UF at d = 13",
            "body1": "BP rounds: 219.1 ms\nUF: 9.9 ms\nInit + projection: 5.0 ms",
            "heading2": "93.6% in BP rounds",
            "body2": "UF stays near 10 ms\nas the BP budget rises.\nBP is the main timing cost.",
            "caption": "Each budget includes its cumulative BP work, its own weight projection, and UF.",
            "alt": "Stacked serial time at distance 13 for BP1, BP2, BP5 and BP10 plus UF. Totals are 35.7, 58.5, 124.2 and 234.0 ms. BP rounds dominate as the budget increases; UF costs about 9.4 to 9.9 ms.",
        },
    ]
    plan = {"presentation_id": deck["presentationId"], "source_revision": deck["revisionId"],
            "source_order": [s["objectId"] for s in deck["slides"]],
            "exemplar_id": exemplar_id, "layout_id": "p10", "slides": specs,
            "media_mapping": {"inherited_UW_logo_and_gold_rule": "keep", "exemplar_chart_element_10": "replace with corresponding 16:9 figure; preserve aspect ratio"},
            "geometry": "Duplicate the exemplar. Preserve title, image, slide number, layout and master. Adjust right-column vertical slots to fit 3–4 lines at the existing 14pt body / 16pt heading styles. Add a 10pt native caption at the bottom left.",
            "original_text_elements": {k: v.get("shape", {}).get("text") for k, v in elements.items() if "shape" in v}}
    (workspace / "slide-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    requests = []
    for spec in specs:
        sid = spec["id"]
        mapping = {exemplar_id: sid}
        mapping.update({e["objectId"]: sid + "_" + e["objectId"].rsplit("_", 1)[-1] for e in exemplar["pageElements"]})
        requests.append({"duplicateObject": {"objectId": exemplar_id, "objectIds": mapping}})
        for key, value in [("1", spec["title"]), ("4", spec["heading1"]), ("5", spec["body1"]), ("6", spec["heading2"]), ("7", spec["body2"])]:
            oid = sid + "_" + key
            # All exemplar text runs within each edited shape have the same style.
            old_styles = [t["textRun"]["style"] for t in elements[key]["shape"]["text"]["textElements"] if "textRun" in t]
            assert all(s == old_styles[0] for s in old_styles)
            style = copy.deepcopy(old_styles[0])
            requests.extend([
                {"deleteText": {"objectId": oid, "textRange": {"type": "ALL"}}},
                {"insertText": {"objectId": oid, "insertionIndex": 0, "text": value}},
                {"updateTextStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": style, "fields": ",".join(style)}}
            ])
            if key != "1":
                transform = copy.deepcopy(elements[key]["transform"])
                if spec["figure"] != "relative_accuracy":
                    y, height = {"4": (133, 29), "5": (164, 80), "6": (252, 29), "7": (283, 67)}[key]
                    transform["translateY"] = y * 12700
                    transform["scaleY"] = height * 12700 / 3000000
                requests.append({"updatePageElementTransform": {"objectId": oid, "applyMode": "ABSOLUTE", "transform": transform}})
        cap_id = sid + "_caption"
        requests.extend([
            {"createShape": {"objectId": cap_id, "shapeType": "TEXT_BOX", "elementProperties": {
                "pageObjectId": sid, "size": {"width": {"magnitude": 558, "unit": "PT"}, "height": {"magnitude": 18, "unit": "PT"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": 23, "translateY": 379, "unit": "PT"}}}},
            {"insertText": {"objectId": cap_id, "text": spec["caption"], "insertionIndex": 0}},
            {"updateTextStyle": {"objectId": cap_id, "textRange": {"type": "ALL"}, "style": {
                "fontFamily": "Arial", "fontSize": {"magnitude": 10, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": {"red": 0.33, "green": 0.33, "blue": 0.33}}}},
                "fields": "fontFamily,fontSize,foregroundColor"}},
            {"updatePageElementAltText": {"objectId": sid + "_10", "title": spec["title"], "description": spec["alt"]}}
        ])
    (workspace / "create-slides-requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    print(json.dumps({"slides": len(specs), "requests": len(requests), "new_slide_ids": [s["id"] for s in specs]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    main(p.parse_args().workspace)
