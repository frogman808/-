from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .inference import rollout
from .planner import plan_task
from .train_policy import CHECKPOINT_PATH


HOST = "127.0.0.1"
PORT = 8000


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Robot Arm Subgoal Demo</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f4f6f8; color: #17202a; }
    main { display: grid; grid-template-columns: minmax(420px, 1fr) 380px; gap: 18px; padding: 18px; height: 100vh; box-sizing: border-box; }
    .stage { background: #ffffff; border: 1px solid #d9e0e7; border-radius: 8px; padding: 14px; display: grid; grid-template-rows: auto 1fr; min-height: 0; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    .status { font-size: 13px; color: #55616f; }
    canvas { width: 100%; height: 100%; min-height: 420px; background: #eef3f7; border: 1px solid #cbd5df; border-radius: 6px; }
    aside { display: grid; grid-template-rows: auto auto 1fr; gap: 12px; min-height: 0; }
    section { background: #ffffff; border: 1px solid #d9e0e7; border-radius: 8px; padding: 12px; }
    textarea { width: 100%; height: 88px; resize: vertical; box-sizing: border-box; border: 1px solid #b8c4d1; border-radius: 6px; padding: 10px; font: inherit; }
    button { height: 38px; border: 0; border-radius: 6px; background: #1f6feb; color: #fff; font-weight: 650; cursor: pointer; padding: 0 14px; }
    button:disabled { background: #8aa9d6; cursor: default; }
    .controls { display: flex; gap: 8px; margin-top: 10px; }
    ol { margin: 8px 0 0 22px; padding: 0; font-size: 13px; line-height: 1.55; }
    pre { margin: 0; white-space: pre-wrap; font-size: 12px; line-height: 1.4; overflow: auto; max-height: 100%; }
    .log { min-height: 0; overflow: hidden; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; height: auto; } canvas { height: 460px; } }
  </style>
</head>
<body>
<main>
  <div class="stage">
    <div class="topbar">
      <h1>Top-View Camera Simulation</h1>
      <div class="status" id="status">ready</div>
    </div>
    <canvas id="view" width="900" height="620"></canvas>
  </div>
  <aside>
    <section>
      <textarea id="command">A beaker content to B beaker, then pick up the glass rod, stir B for 5 seconds, and return to the initial pose.</textarea>
      <div class="controls">
        <button id="planBtn">Plan</button>
        <button id="runBtn">Run</button>
      </div>
    </section>
    <section>
      <strong>LLM Subgoals</strong>
      <ol id="plan"></ol>
    </section>
    <section class="log">
      <strong>Execution Trace</strong>
      <pre id="log"></pre>
    </section>
  </aside>
</main>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const planEl = document.getElementById('plan');
const logEl = document.getElementById('log');
const commandEl = document.getElementById('command');
let frames = [];
let plan = [];
let anim = 0;

const objects = {
  beakerA: {x: 220, y: 270, label: 'A'},
  beakerB: {x: 650, y: 285, label: 'B'},
  rod: {x: 450, y: 440},
  home: {x: 450, y: 105}
};

function armPoint(joints) {
  const yaw = (joints.base_yaw || 0) * Math.PI / 180;
  const reach = 220 + ((joints.elbow_pitch || 0) - 65) * 1.2;
  return {
    x: objects.home.x + Math.sin(yaw) * reach,
    y: objects.home.y + Math.cos(yaw) * reach + 120
  };
}

function draw(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#e8eef4';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#c3ced9';
  for (let x = 60; x < canvas.width; x += 80) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
  for (let y = 60; y < canvas.height; y += 80) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }

  drawBeaker(objects.beakerA.x, objects.beakerA.y, 'A', '#f6c85f');
  drawBeaker(objects.beakerB.x, objects.beakerB.y, 'B', '#76b7b2');
  drawRod(objects.rod.x, objects.rod.y);
  drawRobot(frame);
}

function drawBeaker(x, y, label, liquid) {
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.ellipse(x, y, 58, 42, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = liquid;
  ctx.beginPath(); ctx.ellipse(x, y + 8, 44, 24, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#1f2937'; ctx.font = '700 22px Segoe UI'; ctx.fillText(label, x - 8, y + 7);
}

function drawRod(x, y) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(-0.55);
  ctx.fillStyle = '#667085'; ctx.fillRect(-85, -6, 170, 12);
  ctx.restore();
}

function drawRobot(frame) {
  const joints = frame?.joints || {base_yaw: 0, shoulder_pitch: -42, elbow_pitch: 84, wrist_pitch: -42, wrist_roll: 0, gripper: 62};
  const p = armPoint(joints);
  ctx.strokeStyle = '#1f2937'; ctx.lineWidth = 10; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(objects.home.x, objects.home.y); ctx.lineTo((objects.home.x + p.x) / 2, (objects.home.y + p.y) / 2); ctx.lineTo(p.x, p.y); ctx.stroke();
  ctx.fillStyle = '#111827'; ctx.beginPath(); ctx.arc(objects.home.x, objects.home.y, 20, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(p.x, p.y, 13, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#111827'; ctx.font = '13px Segoe UI';
  ctx.fillText(frame?.subgoal || 'waiting for command', 24, 34);
}

async function requestJson(path, body) {
  const res = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function planOnly() {
  statusEl.textContent = 'planning';
  const data = await requestJson('/api/plan', {command: commandEl.value});
  plan = data.plan;
  planEl.innerHTML = plan.map(s => `<li>${s.name}(${s.args.join(', ')})</li>`).join('');
  statusEl.textContent = 'plan ready';
}

async function run() {
  statusEl.textContent = 'running policy';
  const data = await requestJson('/api/run', {command: commandEl.value});
  plan = data.plan; frames = data.frames;
  planEl.innerHTML = plan.map(s => `<li>${s.name}(${s.args.join(', ')})</li>`).join('');
  logEl.textContent = `frames: ${frames.length}\\nfinal_joints:\\n` + JSON.stringify(data.final_joints, null, 2);
  animate(0);
}

function animate(i) {
  if (!frames.length) { draw(null); return; }
  draw(frames[i]);
  statusEl.textContent = `${i + 1}/${frames.length} ${frames[i].subgoal}`;
  if (i + 1 < frames.length) anim = requestAnimationFrame(() => animate(i + 1));
  else statusEl.textContent = 'finished';
}

document.getElementById('planBtn').onclick = () => planOnly().catch(err => statusEl.textContent = err.message);
document.getElementById('runBtn').onclick = () => run().catch(err => statusEl.textContent = err.message);
draw(null);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        command = payload.get("command", "")
        if self.path == "/api/plan":
            self._send_json({"plan": [step.__dict__ for step in plan_task(command)]})
            return
        if self.path == "/api/run":
            if not CHECKPOINT_PATH.exists():
                self._send_json({"error": "policy checkpoint is missing; run python -m submission.train_policy first"}, 409)
                return
            self._send_json(rollout(command, max_steps_per_subgoal=10))
            return
        self.send_error(404)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"web demo: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

