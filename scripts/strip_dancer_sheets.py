"""Remove individual-dancer worksheet parts before the artifact-tool refresh.

The artifact-tool API does not expose worksheet deletion. This bounded OOXML
operation retains all other parts; artifact-tool performs the subsequent edits
and final export, including rebuilding the shared-string table.
"""

import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(sys.argv[1])
source = root / "dancer_feature_registry.xlsx"
target = root / "dancer_registry_source.xlsx"
main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
with ZipFile(source) as archive:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    types = ET.fromstring(archive.read("[Content_Types].xml"))
    sheets = workbook.find(main + "sheets")
    removed_ids, removed_parts, retained = set(), set(), {}
    for old_index, sheet in enumerate(list(sheets)):
        if re.match(r"^D\d+-", sheet.attrib["name"]):
            removed_ids.add(sheet.attrib[rel_ns + "id"])
            sheets.remove(sheet)
        else:
            retained[old_index] = len(retained)
    for rel in list(rels):
        if rel.attrib["Id"] in removed_ids:
            part = rel.attrib["Target"]
            removed_parts.add(part.lstrip("/") if part.startswith("/")
                              else posixpath.normpath("xl/" + part))
            rels.remove(rel)
    for entry in list(types):
        if entry.attrib.get("PartName", "").lstrip("/") in removed_parts:
            types.remove(entry)
    names = workbook.find(main + "definedNames")
    if names is not None:
        for name in list(names):
            if "localSheetId" in name.attrib:
                old_index = int(name.attrib["localSheetId"])
                if old_index not in retained:
                    names.remove(name)
                else:
                    name.attrib["localSheetId"] = str(retained[old_index])
    replacements = {"xl/workbook.xml": workbook, "xl/_rels/workbook.xml.rels": rels,
                    "[Content_Types].xml": types}
    def serialize(element):
        # OPC requires an unprefixed Types root; preserve each part's default namespace.
        ET.register_namespace("", element.tag[1:].split("}")[0])
        ET.register_namespace("r", rel_ns[1:-1])
        return ET.tostring(element, encoding="utf-8", xml_declaration=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as output:
        for entry in archive.infolist():
            if entry.filename in removed_parts:
                continue
            data = (serialize(replacements[entry.filename])
                    if entry.filename in replacements else archive.read(entry.filename))
            output.writestr(entry, data)
print(f"Removed {len(removed_ids)} individual-dancer worksheets")
