// Refresh the existing registry artifact while preserving its retained sheet layouts.
import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [root, mode] = process.argv.slice(2);
const file = path.join(root, 'dancer_feature_registry.xlsx');
const source = mode === 'inspect' ? file : path.join(root,'dancer_registry_source.xlsx');
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const previewDir = path.join(root, 'registry_previews');
await fs.mkdir(previewDir, {recursive:true});
if (mode === 'inspect') {
  for (const [name, range] of [['Registry Summary','A1:F22'],['Labels-20261015-230822','A1:L13']]) {
    const image = await wb.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(previewDir,`${name}-before.png`),new Uint8Array(await image.arrayBuffer()));
  }
  console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:2500})).ndjson);
  process.exit(0);
}
const model = JSON.parse(await fs.readFile(path.join(root,'feature_registry_workbook_payload.json'),'utf8'));
const summary = wb.worksheets.getItem('Registry Summary');
summary.getRange('B4:B8').values = [[model.labelled_tracks],[model.dancers],[model.non_dancers],[model.runs.length],[model.generated_at]];
summary.getRange('B8').setNumberFormat('yyyy-mm-dd hh:mm:ss');
summary.getRange('A2').values = [['Saved Dancers labels update both registries. Compare confirms original predictions when no reviewed labels have been saved.']];
summary.getRange('A11:F15').values = model.selected_features.map(x=>[x.rank,x.feature_name,x.category,x.standardized_coefficient,x.absolute_weight,x.direction]);
const inventory = wb.worksheets.getItem('Feature Inventory Summary');
inventory.getRange('B5:B8').values = [[model.labelled_tracks],[model.dancers],[model.non_dancers],[model.selected_features.length]];
inventory.getRange('A12:E16').values = model.selected_features.map(x=>[x.rank,x.feature_name,x.category,x.absolute_weight,x.standardized_coefficient]);
const catalog = wb.worksheets.getItem('Feature Catalog');
const rank = new Map(model.selected_features.map(x=>[x.feature_id,x.rank]));
catalog.getRange(`I5:I${4+model.feature_catalog.length}`).values = model.feature_catalog.map(x=>[rank.get(x.id)??null]);
for (const run of model.runs_detail) {
  const sheet = wb.worksheets.getItem(`Labels-${run.run}`);
  const end = Math.max(5,sheet.getUsedRange().values.length,4+run.tracks.length);
  sheet.getRange(`A5:G${end}`).clear({applyTo:'contents'});
  const corrected = new Set(run.validation.corrected_track_ids || []);
  const status = run.validation.status || (run.labelled_track_count ? 'historical_labels':'awaiting_checklist');
  if (run.tracks.length) sheet.getRange(`A5:G${4+run.tracks.length}`).values = run.tracks.map(t=>[
    t.track_id,t.frames_present,t.reviewed_label||'',t.predicted_label,t.dancer_probability,
    status.replaceAll('_',' '),corrected.has(t.track_id)?'Yes':'']);
  sheet.getRange('I5:L9').values = model.selected_features.map(x=>[
    x.feature_name,x.standardized_coefficient,x.absolute_weight,x.direction]);
}
wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson);
const temp = path.join(root,'dancer_feature_registry.updated.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(temp);
await fs.rename(temp,file);
await fs.unlink(source);
for (const [name,range] of [['Registry Summary','A1:T23'],['Feature Inventory Summary','A1:H20'],['Feature Catalog','A1:I10'],
    ...model.runs_detail.map(r=>[`Labels-${r.run}`,`A1:L${Math.max(12,r.tracks.length+4)}`])]) {
  const image = await wb.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(previewDir,`${name}.png`),new Uint8Array(await image.arrayBuffer()));
}
console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:2500})).ndjson);
