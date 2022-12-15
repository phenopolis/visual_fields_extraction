"""
Use AWS rekognition detect_text engine
"""
import io
from cmath import isclose
from collections import defaultdict
from operator import attrgetter

import boto3
from hvf_extraction_script.utilities.regex_utils import Regex_Utils
from PIL import Image


class RekognitionText:
    def __init__(self, text_data):
        self.text = text_data.get("DetectedText")
        self.kind = text_data.get("Type")
        self.id = text_data.get("Id")
        self.parent_id = text_data.get("ParentId")
        self.confidence = text_data.get("Confidence")
        self.geometry = text_data.get("Geometry")

    def __repr__(self) -> str:
        return f"{self.id}:{self.parent_id}:{self.kind}:{self.text}"


def columnise(texts, rel_tol=0.02) -> str:
    tops = [x.geometry["BoundingBox"]["Top"] for x in texts]
    ntops = tops[:]
    for n, top in enumerate(tops[:-1]):
        if isclose(top, tops[n + 1], rel_tol=rel_tol):
            ntops.remove(top)
    dic_lines = defaultdict(list)
    for n, top in enumerate(ntops):
        for block in texts:
            ltop = block.geometry["BoundingBox"]["Top"]
            if isclose(top, ltop, rel_tol=rel_tol):
                dic_lines[n].append(block)
            elif ltop > top:
                break

    # sort by Left within line
    res = []
    for oo in [sorted(x, key=lambda y: y.geometry["BoundingBox"]["Left"]) for x in dic_lines.values()]:  # type: ignore
        res.append(" ".join(list(map(attrgetter("text"), oo))))

    return "\n".join(res)


class Ocr_Utils:
    @staticmethod
    def perform_ocr(img_arr, proc_img: bool = False, column: bool = True, debug_dir: str = "") -> str:
        img = Image.fromarray(img_arr)
        client = boto3.client("rekognition")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()
        res = client.detect_text(Image={"Bytes": data})
        texts = [x for x in [RekognitionText(text) for text in res["TextDetections"]] if x.kind == "LINE"]
        if column:
            text = columnise(texts)
        else:
            text = " ".join(
                [x.text for x in [RekognitionText(text) for text in res["TextDetections"]] if x.kind == "LINE"]
            )

        if debug_dir:
            out = Regex_Utils.temp_out(debug_dir=debug_dir)
            img.save(f"{out}.jpg")
            with open(f"{out}.txt", "w") as f:
                f.writelines(text)

        # Return extracted text:
        return text
