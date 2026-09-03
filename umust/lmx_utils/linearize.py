import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from .Linearizer import Linearizer
from .Delinearizer import Delinearizer
from .symbolic.MxlFile import MxlFile
from .symbolic.part_to_score import part_to_score


def linearize(filename: str):
    if filename == "-":
        input_xml = sys.stdin.readline()
        mxl = MxlFile(ET.ElementTree(
            ET.fromstring(input_xml))
        )
    elif filename.endswith(".mxl"):
        mxl = MxlFile.load_mxl(filename)
    else:
        with open(filename, "r") as f:
            input_xml = "\n".join(f.readlines())
            mxl = MxlFile(ET.ElementTree(
                ET.fromstring(input_xml))
            )
    
    try:
        part = mxl.get_piano_part()
    except:
        part = mxl.tree.find("part")
    
    if part is None or part.tag != "part":
        print("No <part> element found.", file=sys.stderr)
        exit()
    
    linearizer = Linearizer(
        errout=sys.stderr
    )
    linearizer.process_part(part)
    output_lmx = " ".join(linearizer.output_tokens)
    
    return output_lmx
    if filename == "-":
        print(output_lmx)
    else:
        out_path = Path(str(Path(filename).parent).replace("score_xml", "score_lmx")) / (Path(filename).stem + ".lmx")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            print(output_lmx, file=f)