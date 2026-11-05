import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [root, moduleRoot] = process.argv.slice(2);
const require = createRequire(path.join(moduleRoot || process.cwd(), '_training_loader.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(path.join(root, 'training_workbook_payload.json'), 'utf8'));
const { state, features, model, runs } = payload;
const featureNames = new Map(features.map(feature => [feature.id, feature.name]));
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

const overview = base('LASSO features');
title(overview, 'LASSO feature selection and LSTM training cohort');
write(overview, 3, [['Target', state.target], ['Validated videos', runs.length],
  ['Reviewed tracks', state.reviewed_tracks || 0], ['Status', state.status],
  ['Classifier target', 'dancer / non_dancer']]);
overview.getRange('E3:F4').values = [['Updated (UTC)', state.updated_at || null],
  ['Saved revision', payload.database_snapshot ?? null]];
overview.getRange('F3:H3').merge();
overview.getRange('F3').setNumberFormat('yyyy-mm-dd hh:mm:ss');
overview.getRange('A9:H10').merge();
overview.getRange('A9').values = [['Only the first 15 fully validated video runs enter this cohort. LASSO selects five pose features; the LSTM uses those five signals for dancer probability. After the cohort is trained, this registry remains frozen.']];
overview.getRange('A9:H10').format.wrapText = true;
overview.getRange('A9:H10').format.fill = pale;
header(overview, 12, ['Rank', 'Selected feature', 'Feature id', 'LASSO ranking weight', 'Five-input coefficient', 'Direction']);
for (let i=0; i<5; i++) {
  const id = model?.feature_ids[i];
  const weight = model?.coefficient[i] ?? null;
  write(overview, 13+i, [[i+1, id ? featureNames.get(id) || id : 'Awaiting reviewed data',
    id ?? null, model?.ranking_coefficients[i] ?? null, weight,
    weight === null ? null : weight >= 0 ? 'more dancer-like' : 'more non-dancer-like']]);
}
overview.getRange('D13:E17').setNumberFormat('0.0000');
overview.getRange('B13:C17').format.wrapText = true;
overview.getRange('A13:F17').format.rowHeight = 34;
overview.getRange('A19:H20').merge();
overview.getRange('A19').values = [['LASSO selects the five features listed above. The LSTM uses those five measurements in nine overlapping 3-second windows within the middle 5 seconds. Reviewed dancer/non_dancer labels are the training targets.']];
overview.getRange('A19:H20').format.wrapText = true;
overview.getRange('A22:H23').merge();
overview.getRange('A22').values = [[`LSTM fit: ${state.lstm_runs?.length || 0} validated videos and ${state.lstm_samples || 0} usable reviewed sequences. The model updates until 15 validated videos are enrolled. No independent test accuracy is reported.`]];
overview.getRange('A22:H23').format.wrapText = true;
header(overview, 25, ['Video slot', 'Run', 'Track count', 'Reviewed count', 'Window start (s)', 'Window end (s)', 'Window status']);
const runRows = Array.from({length: state.target}, (_, i) => {
  const r = runs[i];
  return [i+1, r?.run || 'Awaiting validated video', r?.tracks.length ?? null,
    r?.tracks.filter(t => t.manual_label).length ?? null, r?.crop.start_s ?? null,
    r?.crop.end_s ?? null, r ? (r.crop.error || 'Extracted') : 'Awaiting video'];
});
write(overview, 26, runRows);
widths(overview, [['A',24],['B',44],['C',23],['D',25],['E',18],['F',18],['G',30],['H',18]]);
overview.freezePanes.freezeRows(1);

const lstm = base('LSTM predictions', 8);
title(lstm, 'LSTM dancer predictions');
lstm.getRange('A3:H4').merge();
lstm.getRange('A3').values = [['The LSTM uses exactly the five LASSO-selected features over time, with a 0.5 dancer threshold. Training fit means the track helped train the model; it is not independent validation. Blank probabilities have a reason in Model status.']];
lstm.getRange('A3:H4').format.wrapText = true;
lstm.getRange('A3:H4').format.rowHeight = 30;
const predictionRows = runs.flatMap(r => r.tracks.map(t => [r.run, t.track_id,
  t.lstm_prediction?.predicted_label ?? null, t.lstm_prediction?.probability ?? null,
  t.manual_label || null, t.lstm_status || 'Awaiting current LSTM model',
  t.lstm_prediction?.fingerprint ?? null, state.updated_at || null]));
table(lstm, 6, ['Run','Track id','LSTM prediction','Dancer probability','Validation label','Model status','Model version','Updated (UTC)'], predictionRows, 'LSTMPredictions');
if (predictionRows.length) {
  lstm.getRange(`D7:D${6+predictionRows.length}`).setNumberFormat('0.0%');
  lstm.getRange(`H7:H${6+predictionRows.length}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
  lstm.getRange(`F7:F${6+predictionRows.length}`).format.wrapText = true;
  lstm.getRange(`A7:H${6+predictionRows.length}`).format.rowHeight = 36;
  lstm.getRange(`G7:G${6+predictionRows.length}`).format.font.size = 8;
  lstm.getRange(`G7:G${6+predictionRows.length}`).format.wrapText = true;
}
widths(lstm, [['A',28],['B',12],['C',22],['D',22],['E',22],['F',43],['G',45],['H',36]]);
lstm.freezePanes.freezeRows(6); lstm.freezePanes.freezeColumns(2);

const history = base('Weight history', 12);
title(history, 'LASSO feature weight history', 12);
history.getRange('A3:L4').merge();
history.getRange('A3').values = [['Each model version retains its selected features, ranking weights and five-input coefficients. Filter by feature to follow its weight across updates. History starts with the database migration.']];
history.getRange('A3:L4').format.wrapText = true;
const weightRows = (payload.weight_history || []).map(w => [w.created_at,w.fingerprint,w.rank,
  w.feature_id,featureNames.get(w.feature_id) || w.feature_id,w.ranking_weight,w.coefficient,
  w.coefficient >= 0 ? 'more dancer-like' : 'more non-dancer-like',w.median,w.mean,w.std,
  w.fingerprint === payload.lstm?.fingerprint ? 'Current LSTM selection' : 'Saved LASSO selection']);
table(history, 6, ['Saved (UTC)','Model version','Rank','Feature id','Feature name','Ranking weight','Model coefficient','Direction','Imputation median','Training mean','Training SD','Selection status'], weightRows, 'WeightHistory');
if (weightRows.length) {
  history.getRange(`F7:G${6+weightRows.length}`).setNumberFormat('0.0000');
  history.getRange(`I7:K${6+weightRows.length}`).setNumberFormat('0.0000');
  history.getRange(`A7:A${6+weightRows.length}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
  history.getRange(`A7:L${6+weightRows.length}`).format.rowHeight = 36;
  history.getRange(`B7:E${6+weightRows.length}`).format.wrapText = true;
  history.getRange(`B7:B${6+weightRows.length}`).format.font.size = 8;
}
widths(history,[['A',36],['B',45],['C',10],['D',42],['E',35],['F',20],['G',22],['H',25],['I',22],['J',20],['K',20],['L',30]]);
history.freezePanes.freezeRows(6);

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
  if (!r) write(s, 7, [[null,null,'Awaiting next validated video']]);
  widths(s,[['A',43],['B',28],['C',35],['D',21],['E',23],['F',28],['G',26],['H',29],['I',24],['J',24],['K',24],['L',24],['M',24]]);
  s.getRange('B3').format.font.size = 8;
  s.freezePanes.freezeRows(6);
}
wb.recalculate();
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1500});
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(wb);
const temporaryWorkbook = path.join(root, 'lasso_lstm_registry.tmp.xlsx');
await output.save(temporaryWorkbook);
await fs.rename(temporaryWorkbook, path.join(root, 'lasso_lstm_registry.xlsx'));
await fs.rm(`${temporaryWorkbook}.inspect.ndjson`, {force: true});
if (process.env.DDC_RENDER_WORKBOOK === '1') {
  const previews = path.join(root, 'workbook_previews'); await fs.mkdir(previews,{recursive:true});
  const requested = process.env.DDC_RENDER_SHEETS?.split(',');
  for (const name of ['LASSO features','LSTM predictions','Weight history','Predictor statistics',...Array.from({length:state.target},(_,i)=>`Run ${String(i+1).padStart(2,'0')}`)]) {
    if (requested && !requested.includes(name)) continue;
    const runIndex = name.startsWith('Run ') ? Number(name.slice(4))-1 : -1;
    const endRow = runIndex >= 0 ? Math.max(7,6+(runs[runIndex]?.tracks.length || 0)) : 55;
    const range = name === 'LASSO features' ? `A1:H${Math.max(43, 27+state.target)}` : name === 'LSTM predictions' ? 'A1:H15' : name === 'Weight history' ? 'A1:L16' : name === 'Predictor statistics' ? 'A1:L12' : `A1:M${endRow}`;
    const preview = await wb.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(previews,`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  console.log((await wb.inspect({kind:'table',range:'LASSO features!A12:F17',include:'values',maxChars:1800})).ndjson);
}
