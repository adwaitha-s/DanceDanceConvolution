"use strict";

const el = id => document.getElementById(id);
let video = el("video");
let run = null;
let frame = 0; // Zero-based in the player, one-based in the CSV.
let ready = false;
let busy = false;
const drafts = new Map();
const rowsByFrame = new Map();

function clock(us) {
  const ms = Math.max(0, Math.round(us / 1000));
  const pad = (n, width = 2) => String(n).padStart(width, "0");
  const h = Math.floor(ms / 3600000);
  const m = Math.floor(ms / 60000) % 60;
  const s = Math.floor(ms / 1000) % 60;
  return (h ? h + ":" : "") + pad(m) + ":" + pad(s) + "." + pad(ms % 1000, 3);
}

function status(message = "", error = false) {
  el("status").textContent = message;
  el("status").classList.toggle("error", error);
}

async function request(path, options) {
  const res = await fetch("/api/runs" + path, options);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : "The request failed. Please try again.");
  }
  return res.json();
}

function buttonIcon(id, icon, label, shortcut = "") {
  const button = el(id);
  button.querySelector("use").setAttribute("href", "#i-" + icon);
  button.setAttribute("aria-label", label);
  button.title = label + shortcut;
}

function controls() {
  const active = ready && !busy;
  const draft = drafts.get(run?.id);
  for (const id of ["play", "volume", "fullscreen", "scrub", "speed", "comment"]) el(id).disabled = !active;
  el("prev").disabled = !active || frame <= 0;
  el("next").disabled = !active || frame >= run.frames.length - 1;
  el("upload").disabled = busy;
  el("runs").disabled = busy;
  el("save").disabled = !active || !draft?.text.trim();
  el("cancel").hidden = !draft?.editing;
  el("cancel").disabled = busy;
  el("save").querySelector("span").textContent = draft?.editing ? "Save changes" : "Save comment";
}

async function whileBusy(action, message = "") {
  if (busy) return;
  busy = true;
  controls();
  status(message);
  try {
    await action();
  } catch (error) {
    status(error.message, true);
    if (run) el("runs").value = run.id;
  } finally {
    busy = false;
    controls();
  }
}

function updateComposer() {
  const draft = drafts.get(run.id);
  const time = clock(run.frames[draft?.frame ?? frame]);
  el("note-time").textContent = "At " + time;
  el("hint").textContent = "Comment will be saved at " + time;
}

function setFrame(index) {
  for (const row of rowsByFrame.get(frame) || []) row.classList.remove("current");
  frame = index;
  for (const row of rowsByFrame.get(frame) || []) row.classList.add("current");
  const us = run.frames[frame];
  el("position").textContent = clock(us);
  el("frame").textContent = "Frame " + (frame + 1);
  el("scrub").value = us;
  el("scrub").setAttribute("aria-valuetext", clock(us) + ", frame " + (frame + 1));
  el("scrub").style.setProperty("--played", (100 * us / el("scrub").max) + "%");
  updateComposer();
  el("prev").disabled = !ready || busy || frame === 0;
  el("next").disabled = !ready || busy || frame === run.frames.length - 1;
}

function frameAt(us) {
  let lo = 0;
  let hi = run.frames.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (run.frames[mid] <= us) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}

function goTo(index) {
  if (!ready || busy) return;
  video.pause();
  const i = Math.max(0, Math.min(index, run.frames.length - 1));
  // Seeking inside the presentation interval avoids rounding onto a neighboring frame.
  const start = run.frames[i];
  const end = run.frames[i + 1] ?? run.end_us;
  video.currentTime = (start + (end - start) / 2) / 1e6;
  setFrame(i);
}

function togglePlay() {
  if (!ready || busy) return;
  if (!video.paused) return goTo(frame);
  if (video.ended) video.currentTime = 0;
  video.play().catch(() => status("Playback could not start. Try pressing Play again.", true));
}

function installVideo() {
  const old = video;
  old.pause();
  video = old.cloneNode(false);
  old.replaceWith(video);
  old.removeAttribute("src");
  old.load();
  const player = video;
  el("player").classList.remove("has-video");
  buttonIcon("play", "play", "Play", " (Space)");
  buttonIcon("volume", "volume", "Mute");
  player.addEventListener("loadeddata", () => {
    if (player !== video) return;
    ready = true;
    el("player").classList.add("has-video");
    controls();
  }, {once: true});
  player.addEventListener("error", () => {
    if (player !== video) return;
    ready = false;
    controls();
    status("Your browser could not play this preview. Open the app in a browser with MP4 playback.", true);
  });
  player.addEventListener("play", () => buttonIcon("play", "pause", "Pause", " (Space)"));
  player.addEventListener("pause", () => buttonIcon("play", "play", "Play", " (Space)"));
  player.addEventListener("ended", () => setFrame(run.frames.length - 1));
  player.addEventListener("click", togglePlay);
  function display(seconds) {
    if (!player.paused && !player.seeking) {
      const i = frameAt(Math.round(seconds * 1e6));
      if (i !== frame) setFrame(i);
    }
  }
  if (player.requestVideoFrameCallback) {
    function presented(_, metadata) {
      if (player !== video) return;
      display(metadata.mediaTime);
      player.requestVideoFrameCallback(presented);
    }
    player.requestVideoFrameCallback(presented);
  } else {
    player.addEventListener("timeupdate", () => display(player.currentTime));
  }
  player.src = "/api/runs/" + run.id + "/video";
  player.playbackRate = Number(el("speed").value);
}

function editNote(note) {
  if (busy || !ready) return;
  goTo(note.frame - 1);
  drafts.set(run.id, {id: note.id, editing: true, frame, text: note.comment});
  el("comment").value = note.comment;
  updateComposer();
  controls();
  el("comment").focus();
}

function showNotes() {
  const notes = el("notes");
  notes.replaceChildren();
  el("markers").replaceChildren();
  rowsByFrame.clear();
  const count = run?.notes.length || 0;
  el("note-count").textContent = count;
  el("comment-count").textContent = count + (count === 1 ? " comment" : " comments");
  if (!count) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No notes yet.";
    notes.append(empty);
  }
  for (const note of run?.notes || []) {
    const i = note.frame - 1;
    const time = clock(run.frames[i]);
    const row = document.createElement("article");
    row.className = "note-row";
    row.dataset.note = note.id;
    const stamp = document.createElement("button");
    stamp.className = "stamp";
    stamp.textContent = time;
    stamp.setAttribute("aria-label", "Jump to " + time + ", frame " + note.frame);
    stamp.onclick = () => goTo(i);
    const body = document.createElement("p");
    body.className = "note-text";
    body.textContent = note.comment;
    const edit = document.createElement("button");
    edit.className = "edit";
    edit.textContent = "Edit comment";
    edit.onclick = () => editNote(note);
    row.append(stamp, body, edit);
    notes.append(row);
    if (!rowsByFrame.has(i)) {
      rowsByFrame.set(i, []);
      const marker = document.createElement("button");
      marker.className = "marker";
      marker.title = "Comment at " + time;
      marker.setAttribute("aria-label", marker.title);
      marker.style.left = (100 * run.frames[i] / el("scrub").max) + "%";
      marker.onclick = () => { goTo(i); row.scrollIntoView({block: "nearest"}); };
      el("markers").append(marker);
    }
    rowsByFrame.get(i).push(row);
    row.classList.toggle("current", i === frame);
  }
}

async function listRuns() {
  const runs = await request("");
  const menu = el("runs");
  const placeholder = new Option(runs.length ? "Select a saved run" : "No saved runs yet", "");
  placeholder.disabled = true;
  menu.replaceChildren(placeholder);
  for (const item of runs) {
    const when = new Date(item.created).toLocaleString();
    menu.add(new Option(item.filename + " · " + when + " · " + item.id.slice(-8), item.id));
  }
  menu.value = run?.id || "";
}

async function openRun(id) {
  video.pause();
  const data = await request("/" + id);
  run = data;
  ready = false;
  frame = 0;
  el("runs").value = id;
  el("filename").textContent = run.filename;
  const size = run.size < 1e6 ? Math.ceil(run.size / 1000) + " KB" : (run.size / 1e6).toLocaleString(undefined, {maximumFractionDigits: 1}) + " MB";
  el("file-info").textContent = run.width + " × " + run.height + "  ·  " + Number(run.fps.toFixed(2)) +
    " fps  ·  " + size;
  el("preview-label").textContent = "Preview · " + run.height + "p";
  el("duration").textContent = clock(run.end_us);
  el("timeline-end").textContent = clock(run.end_us);
  el("csv").href = "/api/runs/" + id + "/notes.csv";
  el("csv").hidden = false;
  el("scrub").max = Math.max(1, run.frames[run.frames.length - 1]);
  el("comment").value = drafts.get(id)?.text || "";
  showNotes();
  setFrame(0);
  installVideo();
  status();
}

el("upload").onclick = () => el("file").click();
el("file").onchange = () => {
  const file = el("file").files[0];
  if (!file) return;
  el("file").value = "";
  video.pause();
  whileBusy(async () => {
    const form = new FormData();
    form.append("video", file);
    const saved = await request("", {method: "POST", body: form});
    await listRuns();
    await openRun(saved.id);
  }, "Preparing " + file.name + "…");
};
el("runs").onchange = () => whileBusy(() => openRun(el("runs").value), "Opening run…");
el("play").onclick = togglePlay;
el("prev").onclick = () => goTo(frame - 1);
el("next").onclick = () => goTo(frame + 1);
el("scrub").oninput = () => goTo(frameAt(Number(el("scrub").value)));
el("speed").onchange = () => { video.playbackRate = Number(el("speed").value); };
el("volume").onclick = () => {
  video.muted = !video.muted;
  buttonIcon("volume", video.muted ? "muted" : "volume", video.muted ? "Unmute" : "Mute");
};
el("fullscreen").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await el("player").requestFullscreen();
  } catch { status("Fullscreen is unavailable in this browser.", true); }
};
el("comment").onfocus = () => { if (!video.paused) goTo(frame); };
el("comment").oninput = () => {
  const draft = drafts.get(run.id) || {id: crypto.randomUUID(), frame};
  draft.text = el("comment").value;
  if (draft.text || draft.editing) drafts.set(run.id, draft);
  else drafts.delete(run.id);
  updateComposer();
  controls();
};
el("cancel").onclick = () => {
  drafts.delete(run.id);
  el("comment").value = "";
  updateComposer();
  controls();
};
el("composer").onsubmit = event => {
  event.preventDefault();
  const draft = drafts.get(run?.id);
  if (!ready || !draft?.text.trim()) return;
  whileBusy(async () => {
    const saved = await request("/" + run.id + "/notes/" + draft.id, {
      method: "PUT", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({frame: draft.frame + 1, comment: draft.text})
    });
    run.notes = saved;
    drafts.delete(run.id);
    el("comment").value = "";
    showNotes();
    rowsByFrame.get(draft.frame)?.at(-1)?.scrollIntoView({block: "nearest"});
    updateComposer();
    status("Comment saved at " + clock(run.frames[draft.frame]) + ".");
  }, "Saving comment…");
};
document.addEventListener("keydown", event => {
  if (!ready || busy || event.altKey) return;
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && event.target === el("comment")) {
    event.preventDefault();
    el("composer").requestSubmit();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.target.matches("textarea,select,input:not([type=range])")) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    goTo(frame + (event.key === "ArrowLeft" ? -1 : 1));
  } else if (event.code === "Space" && !event.target.matches("button")) {
    event.preventDefault();
    togglePlay();
  }
});

showNotes();
whileBusy(listRuns);
