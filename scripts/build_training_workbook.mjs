import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [root, moduleRoot] = process.argv.slice(2);
const require = createRequire(path.join(moduleRoot || process.cwd(), '_training_loader.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(path.join(root, 'training_workbook_payload.json'), 'utf8'));
const { state, features, model, runs } = payload;
const wb = Workbook.create();
const navy = '#17324D', teal = '#147D92', pale = '#E8F5F7', blue = '#185ABD';

function write(sheet, row, values) {
  if (!values.length) return;
  sheet.getRangeByIndexes(row - 1, 0, values.length, values[0].length).values = values;
}
function base(name, cols=8) {
  const sheet = wb.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, 150, cols).format.font.name = 'Aptos';
  sheet.getRangeByIndexes(0, 0, 150, cols).format.font.size = 11;
  sheet.getRangeByIndexes(0, 0, 150, cols).format.columnWidth = 20;
  sheet.getRangeByIndexes(0, 0, 150, cols).format.rowHeight = 23;
  sheet.getRangeByIndexes(0, 0, 150, cols).format.verticalAlignment = 'center';
  sheet.tabColor = teal;
  return sheet;
}
function title(s, text, cols=8) {
  write(s, 1, [[text]]);
  const r = s.getRangeByIndexes(0, 0, 1, cols);
  r.format.font.color = navy; r.format.font.bold = true; r.format.font.size = 14;
  r.format.borders = {bottom:{style:'thin',color:navy}};
  r.format.rowHeight = 34;
}
function header(s, row, values) {
  write(s, row, [values]);
  const r = s.getRangeByIndexes(row - 1, 0, 1, values.length);
  r.format.fill = teal; r.format.font.color = '#FFFFFF';
  r.format.font.bold = true; r.format.wrapText = true; r.format.rowHeight = 42;
  r.format.horizontalAlignment = 'center';
}
function widths(s, specs) {
  specs.forEach(([col, width]) => s.getRange(`${col}:${col}`).format.columnWidth = width);
}
function table(s, row, headers, data, name) {
  header(s, row, headers); write(s, row + 1, data);
  if (data.length) {
    s.tables.add(`A${row}:${String.fromCharCode(64+headers.length)}${row+data.length}`, true, name);
    s.getRangeByIndexes(row, 0, data.length, headers.length).format.rowHeight = 28;
  }
}

const overview = base('Training overview');
title(overview, 'Dancer training cohort');
write(overview, 3, [['Target', state.target], ['Captured videos', runs.length],
  ['Reviewed tracks', state.reviewed_tracks || 0], ['Status', state.status],
  ['Classifier target', 'dancer / non_dancer']]);
overview.getRange('A9:H10').merge();
overview.getRange('A9').values = [['Validation labels follow Dancers: Save reviewed labels records checklist edits. Compare confirms original predictions when no review has been saved. Statistics update as each run and review is added.']];
overview.getRange('A9:H10').format.wrapText = true;
overview.getRange('A9:H10').format.fill = pale;
header(overview, 12, ['Rank', 'Selected feature', 'LASSO ranking weight', 'Five-input model weight']);
for (let i=0; i<5; i++) write(overview, 13+i, [[i+1, model?.feature_ids[i] || 'Awaiting reviewed data',
  model?.ranking_coefficients[i] ?? null, model?.coefficient[i] ?? null]]);
overview.getRange('C13:D17').setNumberFormat('0.0000');
overview.getRange('A19:H20').merge();
overview.getRange('A19').values = [['LASSO ranks full-clip pose statistics. Its score is a linear dancer score, not a probability. LSTM uses middle 5-second skeleton xy and visibility masks, 100 frames at 20 Hz.']];
overview.getRange('A19:H20').format.wrapText = true;
overview.getRange('A22:H23').merge();
overview.getRange('A22').values = [[`Rolling LSTM fit: ${state.lstm_runs?.length || 0} videos and ${state.lstm_samples || 0} usable reviewed sequences. New runs and labels trigger refitting when enough data exists. No independent test accuracy is reported.`]];
overview.getRange('A22:H23').format.wrapText = true;
header(overview, 25, ['Video slot', 'Run', 'Track count', 'Reviewed count', 'Window start (s)', 'Window end (s)', 'Window status']);
const runRows = Array.from({length: state.target}, (_, i) => {
  const r = runs[i];
  return [i+1, r?.run || 'Awaiting next video', r?.tracks.length ?? null,
    r?.tracks.filter(t => t.manual_label).length ?? null, r?.crop.start_s ?? null,
    r?.crop.end_s ?? null, r ? (r.crop.error || 'Extracted') : 'Awaiting video'];
});
write(overview, 26, runRows);
if (state.problems?.length) {
  header(overview, 38, ['Outstanding input']);
  state.problems.forEach((p, i) => {overview.getRange(`A${39+i}:H${39+i}`).merge();
    overview.getRange(`A${39+i}`).values = [[p]];
    overview.getRange(`A${39+i}:H${39+i}`).format.wrapText = true;
    overview.getRange(`A${39+i}:H${39+i}`).format.rowHeight = 34;});
}
widths(overview, [['A',24],['B',44],['C',23],['D',25],['E',18],['F',18],['G',30],['H',18]]);
overview.freezePanes.freezeRows(1);

const registry = base('Feature registry', 8);
title(registry, '40 pose feature candidates');
const featureRows = features.map((f, i) => [i+1, f.id, f.category, f.name, f.definition,
  f.unit, f.predictor_eligible ? 'Yes' : 'No', f.availability]);
table(registry, 3, ['Feature #', 'Feature id', 'Category', 'Feature name', 'Definition', 'Unit', 'LASSO eligible', 'Availability'], featureRows, 'FeatureCatalog');
registry.getRange('A4:H43').format.wrapText = true;
registry.getRange('A4:H43').format.rowHeight = 55;
widths(registry, [['A',12],['B',41],['C',30],['D',36],['E',65],['F',27],['G',18],['H',49]]);
registry.freezePanes.freezeRows(3); registry.freezePanes.freezeColumns(2);

const stats = base('Predictor statistics', 12);
title(stats, 'Per-track predictor statistics', 12);
stats.getRange('A3:L4').merge();
stats.getRange('A3').values = [['Weights and training mean/SD update from the reviewed cohort. Validation labels follow the Dancers tab. Run sheets retain the original predictions shown for review.']];
stats.getRange('A3:L4').format.wrapText = true;
const statsRows = [];
if (model) for (const r of runs) for (const t of r.tracks) model.feature_ids.forEach((fid, i) => {
  const raw = t.values[fid]; const effective = raw ?? model.medians[i];
  statsRows.push([r.run, t.track_id, fid, raw, model.medians[i], model.mean[i], model.std[i],
    (effective-model.mean[i])/model.std[i], model.coefficient[i],
    (effective-model.mean[i])/model.std[i]*model.coefficient[i],t.manual_label || null,t.validation_status]);
});
table(stats, 6, ['Run', 'Track id', 'Feature id', 'Observed value', 'Imputation median', 'Training mean', 'Training SD', 'Standardized value', 'LASSO coefficient', 'Score contribution', 'Validation label', 'Review status'], statsRows, 'PredictorStats');
if (!model) write(stats, 7, [['Awaiting reviewed dancer and non-dancer examples from multiple videos.']]);
if (statsRows.length) stats.getRangeByIndexes(6,3,statsRows.length,7).setNumberFormat('0.0000');
widths(stats, [['A',34],['B',13],['C',43]]); stats.freezePanes.freezeRows(6);

for (let index=0; index<state.target; index++) {
  const r = runs[index];
  const s = base(`Run ${String(index+1).padStart(2,'0')}`, 13);
  title(s, `Video ${index+1} label review`, 13);
  write(s, 2, [['Run', r?.run ?? null], ['Pose signature', r?.signature ?? null]]);
  // These two identifiers are checked on import. Do not use clip order as a track identifier.
  const ids = model?.feature_ids || [];
  const reviewRows = (r?.tracks || []).map(t => [t.track_id,t.frames_present,
    t.prediction?.predicted_label ?? 'Awaiting other reviewed videos',t.prediction?.score ?? null,
    t.manual_label || null,t.validation_status,t.predictor_method,t.review_action,...ids.map(fid => t.values[fid])]);
  table(s, 6, ['Track id','Frames present','Predicted label','Predictor score','Validation label','Review status','Predictor method','Review action',...ids], reviewRows, `Review${index+1}`);
  const last = Math.max(7, 6+reviewRows.length);
  s.getRange(`E7:E${last}`).dataValidation = {rule: {type:'list',values:['dancer','non_dancer']}};
  s.getRange(`E7:E${last}`).format.fill = '#FFF1C1';
  s.getRange(`E7:E${last}`).format.font.color = blue;
  if (reviewRows.length) for (let row=7; row<=last; row++)
    s.getRange(`F${row}`).formulas = [[`=IF(E${row}="","Unreviewed",IF(OR(C${row}="dancer",C${row}="non_dancer"),IF(E${row}=C${row},"Validated correct","Corrected"),"Reviewed without prediction"))`]];
  if (reviewRows.length) s.getRange(`D7:D${last}`).setNumberFormat('0.0000');
  if (ids.length && reviewRows.length) s.getRange(`I7:M${last}`).setNumberFormat('0.0000');
  if (reviewRows.length) s.getRange(`C7:H${last}`).format.wrapText = true;
  if (reviewRows.length) s.getRange(`A7:H${last}`).format.rowHeight = 42;
  if (!r) write(s, 7, [[null,null,'Awaiting next successful analysis']]);
  const summaryStart = last + 4;
  const rows = features.map(f => {
    const a = (r?.tracks || []).map(t => t.values[f.id]).filter(v => typeof v === 'number').sort((a,b)=>a-b);
    const mean = a.length ? a.reduce((a,b)=>a+b,0)/a.length : null;
    const sd = a.length ? Math.sqrt(a.reduce((sum,x)=>sum+(x-mean)**2,0)/a.length) : null;
    const median = a.length ? (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2 : null;
    return [f.id, a.length, mean,sd,median,a.length?a[0]:null,a.length?a.at(-1):null,f.unit];
  });
  table(s, summaryStart, ['Feature id','Measured tracks','Mean','Population SD','Median','Minimum','Maximum','Unit'], rows, `RunStats${index+1}`);
  s.getRangeByIndexes(summaryStart,2,40,5).setNumberFormat('0.0000');
  s.getRangeByIndexes(summaryStart,0,40,8).format.wrapText = true;
  s.getRangeByIndexes(summaryStart,0,40,8).format.rowHeight = 45;
  widths(s,[['A',43],['B',28],['C',35],['D',21],['E',23],['F',28],['G',26],['H',29],['I',24],['J',24],['K',24],['L',24],['M',24]]);
  s.getRange('B3').format.font.size = 8;
  s.freezePanes.freezeRows(6);
}
wb.recalculate();
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1500});
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(root, 'training_registry.tmp.xlsx'));
await fs.rename(path.join(root, 'training_registry.tmp.xlsx'), path.join(root, 'training_registry.xlsx'));
if (process.env.DDC_RENDER_WORKBOOK === '1') {
  const previews = path.join(root, 'workbook_previews'); await fs.mkdir(previews,{recursive:true});
  for (const name of ['Training overview','Feature registry','Predictor statistics',...Array.from({length:state.target},(_,i)=>`Run ${String(i+1).padStart(2,'0')}`)]) {
    const runIndex = name.startsWith('Run ') ? Number(name.slice(4))-1 : -1;
    const endRow = runIndex >= 0 ? Math.max(7,6+(runs[runIndex]?.tracks.length || 0))+44 : 55;
    const range = name === 'Feature registry' ? 'A1:H43' : name === 'Training overview' ? 'A1:H45' : name === 'Predictor statistics' ? 'A1:L12' : `A1:M${endRow}`;
    const preview = await wb.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(previews,`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  console.log((await wb.inspect({kind:'table',range:'Feature registry!A3:D6',include:'values',maxChars:1800})).ndjson);
}
